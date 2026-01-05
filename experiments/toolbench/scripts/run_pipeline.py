"""
ToolBench pipeline runner
- Read pipeline configuration (new_code/toolbench_pipeline_config.json)
- Run baseline / watermark / evaluation / decoding sequentially
- All outputs are written to output/toolbench_predictions/<timestamp>/...

This script is a single example run for 9718 (split=test_instruction/G1_category_9718, task_limit=1), adjustable via config.
"""

import argparse
import datetime
import json
import subprocess
import sys
import os

# Unset proxy to avoid connection refused errors in some environments
for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if key in os.environ:
        del os.environ[key]

from pathlib import Path
from typing import Any, Dict, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override or {})
    return merged


def run_cmd(cmd, cwd=None, project_root: Path = None):
    print(f"[CMD] {' '.join(cmd)} (cwd={cwd or Path.cwd()})")
    env = os.environ.copy()
    if project_root:
        env["PYTHONPATH"] = str(project_root)
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if res.returncode != 0:
        raise SystemExit(f"Command failed with code {res.returncode}: {' '.join(cmd)}")


def resolve_query_file(data_root: Path, split: str) -> Tuple[Path, str]:
    """
    Locate the instruction file following DataLoader's split parsing logic.
    Returns: (query_file_path, test_set_name)
    """
    candidates = [
        data_root / "instruction" / f"{split}_query.json",
        data_root / f"{split}.json",
        data_root / "test_instruction" / f"{split}.json",
    ]
    for c in candidates:
        if c.exists():
            stem = c.stem
            parts = stem.split("_")
            # Remove trailing numeric suffix to restore test set name (e.g., G1_category_9718 -> G1_category)
            if parts and parts[-1].isdigit():
                test_set = "_".join(parts[:-1])
            else:
                test_set = stem
            return c, test_set
    raise FileNotFoundError(f"Cannot find instruction file for split={split}, data_root={data_root}")


def run_experiment(cfg: Dict[str, Any], temp_cfg_path: Path, project_root: Path) -> Path:
    temp_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    # run_experiment.py is now in experiments/toolbench/scripts/
    script_path = project_root / "experiments" / "toolbench" / "scripts" / "run_experiment.py"
    run_cmd(
        [sys.executable, "-u", str(script_path), "--config", str(temp_cfg_path)],
        cwd=project_root,
        project_root=project_root
    )
    return Path(cfg["run_name"])


def run_eval(
    eval_cfg: Dict[str, Any],
    pred_dir: Path,
    split: str,
    test_set: str,
    query_file: Path,
    out_root: Path,
    method_name: str,
    project_root: Path,
):
    if not eval_cfg.get("enable"):
        return
    convert_script = Path(eval_cfg["convert_script"])
    eval_script = Path(eval_cfg["eval_script"])

    # 1) Convert predictions to the format required by ToolEval
    converted_root = out_root / "converted"
    model_dir = converted_root / method_name
    model_dir.mkdir(parents=True, exist_ok=True)
    converted_path = model_dir / f"{test_set}.json"

    run_cmd(
        [
            sys.executable,
            "-u",
            str(convert_script),
            "--answer_dir",
            str(pred_dir / split),

            "--method",
            method_name,
            "--output",
            str(converted_path),
        ],
        cwd=project_root,  # Execute from project root
        project_root=project_root
    )

    # 2) Pass rate evaluation
    save_path = out_root / "eval"
    save_path.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable,
            "-u",
            str(eval_script),
            "--converted_answer_path",
            str(converted_root),
            "--save_path",
            str(save_path),
            "--reference_model",
            method_name,
            "--test_ids",
            eval_cfg["test_ids"],
            "--test_set",
            test_set,
            "--max_eval_threads",
            str(eval_cfg.get("max_eval_threads", 5)),
            "--evaluate_times",
            str(eval_cfg.get("evaluate_times", 1)),
            "--overwrite",
        ],
        cwd=project_root,  # Execute from project root
        project_root=project_root
    )


def run_decoding(
    decode_cfg: Dict[str, Any],
    pred_dir: Path,
    run_root: Path,
    pipeline_config: Path,
    project_root: Path,
):
    decode_script = Path(
        decode_cfg.get("script", "experiments/rlnc_trajectory/decode_toolbench_watermark.py")
    ).resolve()
    out_path = run_root / "decode" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable,
            "-u",
            str(decode_script),
            "--pipeline_config",
            str(pipeline_config.resolve()),
            "--pred_dir",
            str(pred_dir),
            "--output",
            str(out_path),
        ],
        cwd=project_root,
        project_root=project_root
    )


def run_analysis(analysis_cfg: Dict[str, Any], run_root: Path, project_root: Path):
    analysis_script = Path(analysis_cfg.get("script", "new_code/analyze_results.py")).resolve()
    run_cmd(
        [
            sys.executable,
            "-u",
            str(analysis_script),
            str(run_root),
        ],
        cwd=project_root,
        project_root=project_root
    )



def resolve_path(path_str: str, root: Path) -> Path:
    """Resolve a path relative to root if it's not absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (root / p).resolve()

def main():
    # Determine project root (this script is in experiments/SynthID_Experiment/scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]  # Go up 3 levels: scripts -> SynthID_Experiment -> experiments -> AgentMark
    
    parser = argparse.ArgumentParser()
    # Default config relative to project root
    default_config = project_root / "experiments" / "toolbench" / "configs" / "pipeline_config.json"
    parser.add_argument("--pipeline_config", default=str(default_config))
    parser.add_argument("--output_root", default=None, help="Root directory for outputs. If not set, uses timestamped dir.")
    parser.add_argument("--split", default=None, help="Specific split to run (overrides config)")
    args = parser.parse_args()

    pipe_cfg = load_json(Path(args.pipeline_config))
    steps = pipe_cfg.get("pipeline_steps", {})
    common_cfg = pipe_cfg.get("common_config", {})
    baseline_specific = pipe_cfg.get("baseline_specific", {})
    watermark_specific = pipe_cfg.get("watermark_specific", {})
    eval_cfg = pipe_cfg.get("eval", {})
    decode_cfg = pipe_cfg.get("decode_config", {})
    
    # Resolve data root
    data_root = resolve_path(common_cfg["data_root"], project_root)
    
    # Export API config to env for subprocesses (especially eval)
    if "base_url" in common_cfg:
        os.environ["OPENAI_BASE_URL"] = common_cfg["base_url"]
    if "api_key" in common_cfg and not os.environ.get("OPENAI_API_KEY"):
         # only set if not already in env (env should take precedence or match config)
         # But usually config has placeholder ${OPENAI_API_KEY} which is not useful to export literally if not expanded.
         # Actually pipeline config loader doesn't expand vars automatically unless we wrote code for it? 
         # Wait, run_experiment.py expands it. run_pipeline.py does NOT seem to expand it in `load_json`.
         # But the common_config["api_key"] is likely "${OPENAI_API_KEY}".
         pass
    
    if "model" in common_cfg:
        os.environ["EVAL_MODEL"] = common_cfg["model"]

    # Determine splits to run
    if args.split:
        splits = [args.split]
        print(f"[INFO] CLI Override: Running only split {splits}")
    else:
        # Get splits list, fallback to a single split if not found
        splits = common_cfg.get("splits", [])
        if not splits:
            splits = [common_cfg.get("split", "G1_instruction")]
        print(f"[INFO] Running splits from config: {splits}")

    if args.output_root:
        run_root = resolve_path(args.output_root, project_root)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Determine prefix based on mode
        is_local = common_cfg.get("use_local_model", False)
        prefix = "local" if is_local else "api"
        dir_name = f"{prefix}_{ts}"
        
        # Output relative to project root
        run_root = (project_root / "output" / "toolbench_predictions" / dir_name).resolve()
    
    run_root.mkdir(parents=True, exist_ok=True)

    def build_cfg(specific: Dict[str, Any], current_split: str) -> Dict[str, Any]:
        cfg = merge_config(common_cfg, specific)
        cfg["split"] = current_split # Override split
        
        run_name = cfg.get("run_name", cfg.get("mode", "run"))
        # Ensure run_name path is absolute and points to run_root
        # Ensure all split results are in the same run_name directory (e.g., exp_fallback/G1/..., exp_fallback/G2/...)
        if not Path(run_name).is_absolute():
            cfg["run_name"] = str((run_root / run_name).resolve())
        
        # Resolve other potential paths in config
        if "bit_stream_path" in cfg:
             cfg["bit_stream_path"] = str(resolve_path(cfg["bit_stream_path"], project_root))
        if "cache_root" in cfg:
             cfg["cache_root"] = str(resolve_path(cfg["cache_root"], project_root))
        if "toolenv_root" in cfg:
             cfg["toolenv_root"] = str(resolve_path(cfg["toolenv_root"], project_root))
        if "data_root" in cfg:
             cfg["data_root"] = str(resolve_path(cfg["data_root"], project_root))
             
        return cfg

    baseline_run_dir = None
    watermark_run_dir = None

    # Iterate through all splits to run experiments and evaluations
    for split in splits:
        print(f"\n{'='*40}\nProcessing Split: {split}\n{'='*40}")
        query_file, test_set = resolve_query_file(data_root, split)

        if steps.get("run_baseline", True):
            baseline_cfg = build_cfg(baseline_specific, split)
            # Temp config in project root for consistency
            tmp_cfg = project_root / "experiments" / "toolbench" / "configs" / f".tmp_baseline_cfg_{split}.json"
            tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
            baseline_run_dir = run_experiment(baseline_cfg, tmp_cfg, project_root)

        if steps.get("run_watermark", True):
            watermark_cfg = build_cfg(watermark_specific, split)
            tmp_cfg = project_root / "experiments" / "toolbench" / "configs" / f".tmp_watermark_cfg_{split}.json"
            tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
            watermark_run_dir = run_experiment(watermark_cfg, tmp_cfg, project_root)

        if steps.get("run_evaluation", True):
            if baseline_run_dir:
                run_eval(
                    eval_cfg,
                    baseline_run_dir,
                    split,
                    test_set,
                    query_file,
                    run_root,
                    "baseline",
                    project_root,
                )
            if watermark_run_dir:
                run_eval(
                    eval_cfg,
                    watermark_run_dir,
                    split,
                    test_set,
                    query_file,
                    run_root,
                    "watermark",
                    project_root,
                )

    # Decoding and Analysis only need to run once as they scan the entire directory
    if steps.get("run_decoding", True) and watermark_run_dir:
        print(f"\n{'='*40}\nRunning Decoding (All Splits)\n{'='*40}")
        # Resolve decode script path
        script_path = decode_cfg.get("script", "experiments/rlnc_trajectory/decode_toolbench_watermark.py")
        decode_cfg["script"] = str(resolve_path(script_path, project_root))
        # Pass watermark_run_dir which contains subdirectories for all splits
        run_decoding(decode_cfg, watermark_run_dir, run_root, Path(args.pipeline_config), project_root)

    if steps.get("run_analysis", True):
        print(f"\n{'='*40}\nRunning Analysis (All Splits)\n{'='*40}")
        analysis_cfg = pipe_cfg.get("analysis_config", {})
        # Resolve analysis script path
        script_path = analysis_cfg.get("script", "scripts/analysis/analyze_results.py")
        analysis_cfg["script"] = str(resolve_path(script_path, project_root))
        run_analysis(analysis_cfg, run_root, project_root)

    print(f"[INFO] pipeline finished. Outputs under {run_root}")


if __name__ == "__main__":
    main()
