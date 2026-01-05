"""
ToolBench/StableToolBench entry point.
- Supports baseline and differential watermark sampling.
- Produces ToolEval-compatible prediction files.
"""

import argparse
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

import sys
import os
# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# moved lazy import inside main


from agentmark.core.parser_utils import extract_and_normalize_probabilities
from agentmark.core.watermark_sampler import sample_behavior, sample_behavior_differential, sample_behavior_red_green
from agentmark.environments.toolbench.adapter import ToolBenchAdapter
from agentmark.environments.toolbench.data_loader import ToolBenchDataLoader
from agentmark.environments.toolbench.output import build_answer_record, save_prediction
from agentmark.environments.toolbench.prompt import build_messages


def expand_vars(obj):
    if isinstance(obj, dict):
        return {k: expand_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_vars(v) for v in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        val = os.environ.get(var_name)
        if val is not None:
             return val
        return obj # Return original if not found (or handle as empty string)
    return obj

def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        config = json.load(f)
    return expand_vars(config)


def load_bit_stream(bit_stream_path: Optional[str]) -> str:
    if bit_stream_path:
        path = Path(bit_stream_path)
        if path.exists():
            return path.read_text().strip()
    # fallback
    random.seed(0)
    return "".join(random.choice("01") for _ in range(4096))


def get_client(api_key: str, base_url: Optional[str]) -> Optional[OpenAI]:
    if api_key and api_key != "YOUR_API_KEY" and OpenAI is not None:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None


def uniform_prob(commands: List[str]) -> Dict[str, float]:
    p = 1.0 / len(commands)
    return {c: p for c in commands}


def parse_action_args(model_output: str, chosen: str) -> Dict:
    """Parse tool arguments from the model output."""
    def _extract_json(text: str) -> Optional[dict]:
        # Handle Thought + JSON outputs by slicing the first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    data = _extract_json(model_output)
    if not isinstance(data, dict):
        return {}

    if isinstance(data, dict):
        action_args = data.get("action_args")
        if isinstance(action_args, dict):
            args = action_args.get(chosen, {})
            if isinstance(args, dict):
                return args
            if isinstance(args, str):
                return {"final_answer": args}

        actions = data.get("actions")
        if isinstance(actions, list):
            for item in actions:
                if isinstance(item, dict) and item.get("tool") == chosen:
                    args = item.get("args", {})
                    if isinstance(args, dict):
                        return args
                    if isinstance(args, str):
                        return {"final_answer": args}

        sai = data.get("selected_action_input")
        if chosen == "Finish" and isinstance(sai, dict):
            return sai
        if chosen == "Finish" and isinstance(sai, str):
            return {"final_answer": sai}
    return {}


def extract_thought(model_output: str) -> str:
    """Try to extract the Thought line from the model output."""
    for line in model_output.splitlines():
        if line.strip().lower().startswith("thought:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_effective_bit_stream(
    base_message: str,
    config: dict,
    num_tasks: int,
) -> str:
    """Build the effective bit stream based on the embedding strategy."""
    if not base_message:
        return base_message

    watermark_config = config.get("watermark_config", {})
    strategy = watermark_config.get("embedding_strategy", "one_time")
    if strategy != "cyclic":
        return base_message

    # ToolBench default max_steps
    max_steps = config.get("max_steps", 6)
    estimated_steps = max(1, num_tasks) * max_steps
    # In differential mode, each step embeds up to log2(n) bits; conservatively estimate 3 bits/step.
    estimated_bits_needed = max(estimated_steps * 3, len(base_message))
    repeats = max(1, (estimated_bits_needed // len(base_message)) + 2)
    effective_stream = base_message * repeats
    print(f"[INFO] Cyclic embedding: estimated {estimated_bits_needed} bits needed, generated {len(effective_stream)} bits (repeats {repeats})")
    return effective_stream


def main():
    print("[DEBUG] main() started")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_toolbench.json", help="Path to the config file.")
    parser.add_argument("--split", type=str, default=None, help="Force a specific split (overrides config).")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific seed (overrides config).")
    parser.add_argument("--run_name", type=str, default=None, help="Force run_name (overrides config).")
    parser.add_argument("--query_id", type=str, default=None, help="Run only the specified query_id (parallel mode).")
    parser.add_argument("--task_index", type=int, default=None, help="Run only the Nth task (parallel mode).")
    parser.add_argument("--sampling_strategy", type=str, default=None, help="Override watermark sampling ('differential', 'red_green').")
    parser.add_argument("--use_rlnc", type=str, default=None, help="Override RLNC setting ('true', 'false').")
    parser.add_argument("--no_watermark", action="store_true", help="Disable watermarking (run as baseline)")
    
    args = parser.parse_args()

    raw_cfg = load_config(Path(args.config))
    print(f"[TRACE] Config loaded, keys: {list(raw_cfg.keys())}")
    
    # Handle hierarchical config (pipeline_config.json style)
    if "common_config" in raw_cfg:
        print("[TRACE] Hierarchical config detected")
        # Defaults
        cfg = raw_cfg["common_config"].copy()
        
        # Determine mode to merge specific config
        current_mode = "baseline"
        if args.sampling_strategy:
            current_mode = "watermark"
        
        if current_mode == "watermark" and "watermark_specific" in raw_cfg:
            cfg.update(raw_cfg["watermark_specific"])
        elif "baseline_specific" in raw_cfg:
             cfg.update(raw_cfg["baseline_specific"])
             
    else:
        print("[TRACE] Flat config detected")
        # Flat config
        cfg = raw_cfg
        
    data_root = Path(cfg["data_root"])
    toolenv_root = Path(cfg["toolenv_root"])
    cache_root = cfg.get("cache_root")
    print(f"[TRACE] Paths resolved. data_root={data_root}")
    
    # CLI Overrides
    split = args.split if args.split else cfg.get("split", "G1")
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    run_name = args.run_name if args.run_name else cfg.get("run_name", f"{cfg.get('mode', 'baseline')}_run")
    
    task_limit = cfg.get("task_limit")
    mode = cfg.get("mode", "baseline")
    max_steps = cfg.get("max_steps", 6)
    temperature = cfg.get("temperature", 0)
    verbose = cfg.get("verbose", False)
    
    run_dir = (
        Path(run_name)
        if Path(run_name).is_absolute()
        else Path("output") / "toolbench_predictions" / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRACE] run_dir ready: {run_dir}")

    filter_failed_ref = cfg.get("filter_failed_reference", False)
    shuffle = cfg.get("shuffle", False)
    
    # Loader Limit Logic:
    # If running parallel specific task, we load ALL then pick one. Limit handled later.
    loader_limit_arg = None if (filter_failed_ref or args.query_id or args.task_index is not None) else task_limit
    print(f"[TRACE] Loader limit arg: {loader_limit_arg}")
    
    watermark_config = cfg.get("watermark_config", {})
    if args.sampling_strategy:
        watermark_config["sampling_strategy"] = args.sampling_strategy
    if args.use_rlnc is not None:
        watermark_config["use_rlnc"] = (args.use_rlnc.lower() == "true")

    if args.no_watermark:
        watermark_config["enable"] = False
        print("Watermarking DISABLED via CLI argument.")

    print(f"[TRACE] Initializing DataLoader with data_root={data_root}, split={split}")
    loader = ToolBenchDataLoader(
        data_root=data_root,
        split=split,
        limit=loader_limit_arg,
        shuffle=shuffle,
        seed=seed,
    )
    print(f"[TRACE] DataLoader initialized. Tasks: {len(loader)}")


    client = None
    if cfg.get("use_local_model", False):
        print(f"[INFO] Using LocalLLMClient with model: {cfg.get('model')}")
        # Extract text watermark config if present
        text_wm_config = cfg.get("text_watermark_config", None) if mode == "watermark" else None
        from agentmark.core.local_llm import LocalLLMClient
        model_path = cfg.get("local_model_path") or cfg.get("model")
        client = LocalLLMClient(
            model_path=model_path,
            device="cuda", # Force CUDA for local run
            torch_dtype="float16",
            watermark_config=text_wm_config
        )
    else:
        print(f"[TRACE] Initializing client for model: {cfg.get('model')}")
        client = get_client(cfg.get("api_key", ""), cfg.get("base_url"))
    
    print(f"[TRACE] Client ready: {client is not None}")
    model = cfg.get("model")
    
    print(f"[TRACE] Initializing Adapter...")
    adapter = ToolBenchAdapter(
        toolenv_root=toolenv_root,
        use_cache=True,
        cache_root=Path(cache_root) if cache_root else None,
        client=client,
        model=model,
        temperature=temperature,
    )
    print(f"[TRACE] Adapter ready.")
    
    raw_bit_stream = load_bit_stream(cfg.get("bit_stream_path"))
    
    # Filtering logic
    if filter_failed_ref:
        print("[INFO] Pre-scanning reference answer directory for 'win' status...")
        prefix = split.split("_")[0] # G1, G2, G3
        answer_dir_name = f"{prefix}_answer"
        ref_dir = data_root / "answer" / answer_dir_name
        
        solved_limit = cfg.get("task_limit", 1000)
        solved_tasks = set()
        
        if ref_dir.exists():
            import glob
            all_ref_files = list(ref_dir.glob("*.json"))
            print(f"[INFO] Found {len(all_ref_files)} reference files. Checking 'win' status...")
            
            for f_path in all_ref_files:
                try:
                    t_id = f_path.name.split("_")[0]
                    with open(f_path, 'r') as f:
                        content = f.read(512)
                        if '"win": true' in content.lower():
                            solved_tasks.add(t_id)
                except:
                    continue
        
        print(f"[INFO] Found {len(solved_tasks)} solved reference tasks.")
        
        filtered_loader = []
        for task in loader:
            t_id = str(task.get("query_id"))
            if t_id in solved_tasks:
                filtered_loader.append(task)
            
            if len(filtered_loader) >= solved_limit:
                break
        
        print(f"[INFO] Filtered {len(loader)} -> {len(filtered_loader)} tasks.")
        loader = filtered_loader

    # --- Task Selection for Parallel Execution ---
    if args.query_id:
        loader = [t for t in loader if str(t.get("query_id")) == str(args.query_id)]
        print(f"[INFO] Precision Selection: Running specific task query_id={args.query_id}")
    elif args.task_index is not None:
        if args.task_index < len(loader):
            loader = [loader[args.task_index]]
            print(f"[INFO] Precision Selection: Running specific task index={args.task_index} (QueryID={loader[0].get('query_id')})")
        else:
            print(f"[WARN] Task index {args.task_index} out of range (Total {len(loader)}). Exiting.")
            loader = []

    # Final Limit Check (if not single task mode)
    if not args.query_id and args.task_index is None and task_limit is not None:
         loader = loader[:task_limit]
         print(f"[INFO] Limited to {len(loader)} tasks (config limit).") 

    use_rlnc = watermark_config.get("use_rlnc", True) # Default Enabled as requested

    # RLNC Setup
    if use_rlnc and mode == "watermark":
        print(f"[INFO] RLNC Enabled. Encoding payload with DeterministicRLNC (key={seed}).")
        from agentmark.core.rlnc_codec import DeterministicRLNC
        # Use simple int seed derived from config seed
        encoder = DeterministicRLNC(raw_bit_stream, stream_key=seed)
        
        # Calculate needed length
        estimated_steps = max(1, len(loader)) * max_steps
        needed_len = max(estimated_steps * 5, 8192) # Generous buffer
        
        bit_stream = encoder.get_stream(0, needed_len)
        print(f"[INFO] Generated RLNC coded stream of length {len(bit_stream)}")
        
        # Save RLNC metadata to run_dir for decoder/verifier usage
        rlnc_meta = {
            "payload": raw_bit_stream,
            "stream_key": seed,
            "is_rlnc": True
        }
        with open(run_dir / "rlnc_meta.json", "w") as f:
            json.dump(rlnc_meta, f)
            
    else:
        bit_stream = build_effective_bit_stream(raw_bit_stream, cfg, len(loader))
    bit_index = 0

    print(f"[INFO] Loaded {len(loader)} tasks from {data_root}, split={split}")
    print(f"[INFO] Mode={mode}, run_dir={run_dir}")
    print(f"[INFO] Watermark Config: {watermark_config}")

    for idx, task in enumerate(loader):
        episode = adapter.prepare_episode(task)
        admissible = episode["admissible_commands"]
        messages = build_messages(
            query=task.get("query", ""),
            tool_summaries=episode["tool_summaries"],
            admissible_commands=admissible,
        )
        # Record task start time
        task_start_time = time.time()
        
        # Interaction loop
        trajectory = []
        step_count = 0
        done = False
        last_observation = episode["observation"]
        final_answer = ""
        chosen_history = []
        trace = [] if mode == "watermark" else None

        while not done and step_count < max_steps:
            # === Model call ===
            model_output = None
            if client and model:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    )
                    model_output = resp.choices[0].message.content
                except Exception as e:  # pragma: no cover
                    print(f"[WARN] model call failed: {e}")

            if not model_output:
                model_output = json.dumps(
                    {"action_weights": uniform_prob(admissible), "action_args": {cmd: {} for cmd in admissible}}
                )

            probs = extract_and_normalize_probabilities(model_output, admissible)
            if not probs:
                probs = uniform_prob(admissible)

            # === Sampling ===
            # If Finish has the highest probability, choose it; otherwise sample among the remaining tools.
            top_tool = max(probs.items(), key=lambda x: x[1])[0] if probs else "Finish"
            effective_probs = probs
            bit_before = bit_index
            if top_tool == "Finish":
                chosen = "Finish"
            else:
                # Renormalize after removing Finish
                effective_probs = {k: v for k, v in probs.items() if k != "Finish"}
                total = sum(effective_probs.values())
                if total <= 0:
                    effective_probs = uniform_prob([k for k in probs.keys() if k != "Finish"])
                else:
                    effective_probs = {k: v / total for k, v in effective_probs.items()}

                if mode == "watermark":
                    # === Strategy Selection ===
                    strategy = watermark_config.get("sampling_strategy", "differential")
                    
                    if strategy == "red_green":
                        # Red-Green List Sampling
                        gamma = watermark_config.get("gamma", 0.5)
                        delta = watermark_config.get("delta", 2.0)
                        chosen, green_list, consumed_bits, _ = sample_behavior_red_green(
                            probabilities=effective_probs,
                            context_for_key=episode["observation"],
                            round_num=idx + step_count,
                            gamma=gamma,
                            delta=delta
                        )
                        # Red-Green doesn't consume bits usually, but interface returns 0
                        # trace extra info?
                        
                    else:
                        # Differential Sampling (Default)
                        chosen, _, consumed_bits, _ = sample_behavior_differential(
                            probabilities=effective_probs,
                            bit_stream=bit_stream,
                            bit_index=bit_index,
                            context_for_key=episode["observation"],
                            round_num=idx + step_count,
                        )
                    
                    bit_index += consumed_bits
                else:
                    chosen = sample_behavior(effective_probs, seed=seed, round_num=idx + step_count)
            # Record watermark/decoding trace
            if trace is not None:  # Only record in watermark mode
                trace.append(
                    {
                        "task_idx": idx,
                        "round": step_count,
                        "mode": mode,
                        "strategy": watermark_config.get("sampling_strategy", "differential"), # Log Strategy
                        "raw_probs": {k: v for k, v in probs.items() if v},
                        "effective_probs": {k: v for k, v in effective_probs.items() if v},
                        "chosen": chosen,
                        "history_tools": list(chosen_history),
                        "context_for_key": episode["observation"],
                        "bit_index_before": bit_before,
                        "bit_index_after": bit_index,
                    }
                )
            chosen_history.append(chosen)

            action_args = parse_action_args(model_output, chosen)
            action = {"tool": chosen, "arguments": action_args}
            trajectory.append({"role": "assistant", "message": model_output, "next": []})

            # Finish ends the episode
            if chosen == "Finish":
                if isinstance(action_args, dict):
                    final_answer = action_args.get("final_answer", "")
                elif isinstance(action_args, str):
                    final_answer = action_args
                done = True
                break

            # Tool execution
            step_result = adapter.step(action, episode["tool_summaries"], state=task)
            trajectory.append({"role": "tool", "message": step_result["observation"], "next": []})
            last_observation = step_result["observation"]
            done = step_result.get("done", False)
            step_count += 1
            if done:
                if not final_answer:
                    final_answer = last_observation
                break

            # Feed tool response back to the model for the next decision
            thought = extract_thought(model_output)
            formatted_action = (
                f"Thought: {thought}\n"
                f"Action: {chosen}\n"
                f"Action Input: {json.dumps(action_args, ensure_ascii=False)}"
            )
            messages.append({"role": "assistant", "content": formatted_action})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation:\n{last_observation}\nContinue Thought/Action/Action Input.",
                }
            )

        if not final_answer:
            final_answer = last_observation

        # Wrap final_answer for evaluation (ToolBench-compatible format)
        if isinstance(final_answer, dict) and {"return_type", "final_answer"} <= set(final_answer.keys()):
            wrapped_answer = final_answer
        else:
            wrapped_answer = {"return_type": "give_answer", "final_answer": final_answer}

        task_end_time = time.time()
        duration = task_end_time - task_start_time

        record = build_answer_record(
            method=mode,
            final_answer=wrapped_answer,
            trajectory=trajectory,
            total_steps=len(trajectory),
            query=task.get("query", ""),
            available_tools=episode.get("tool_summaries", []),
            duration=duration,
            watermark_trace=trace if mode == "watermark" else None,
        )
        query_id = str(task.get("query_id", f"task_{idx}"))
        out_path = save_prediction(run_dir, split, query_id, record)
        print(f"[INFO] saved prediction -> {out_path}")

    print("[INFO] run finished")


if __name__ == "__main__":
    main()
