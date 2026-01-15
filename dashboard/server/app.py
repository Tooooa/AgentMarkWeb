
import json
import re
import uuid
import time
import os
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, AsyncOpenAI
from sentence_transformers import SentenceTransformer, util
import copy
import sys

# --- Path Setup ---
# Must happen before importing from dashboard.*
_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_LOCAL_PROJECT_ROOT))

# --- Shared Utilities ---
from dashboard.server.shared import (
    PROJECT_ROOT, SWARM_ROOT, TOOL_DATA_ROOT,
    _load_root_dotenv, _resolve_api_key, _build_proxy_client,
    _create_dynamic_swarm_tools, _get_base_llm_base, _get_proxy_base
)
from dashboard.server.routers import add_agent

if SWARM_ROOT.exists():
    sys.path.append(str(SWARM_ROOT))

from agentmark.core.rlnc_codec import DeterministicRLNC
from agentmark.core.watermark_sampler import sample_behavior_differential
from agentmark.sdk.prompt_adapter import extract_json_payload, get_prompt_instruction

# --- Database Setup ---
from dashboard.server.database import ConversationDB
db = ConversationDB(db_path=str(PROJECT_ROOT / "dashboard/data/conversations.db"))

# --- Retriever Setup ---
from dashboard.server.retriever import ToolBenchRetriever
from agentmark.environments.toolbench.adapter import ToolBenchAdapter

retriever = None
retriever_loading = False

async def init_retriever():
    global retriever, retriever_loading
    retriever_loading = True
    print("[INFO] Background: Initializing ToolBench Retriever on CPU...")
    try:
        # Run in thread to avoid blocking simple init
        r = await asyncio.to_thread(ToolBenchRetriever, TOOL_DATA_ROOT, device="cpu")
        await asyncio.to_thread(r.load_model)
        await asyncio.to_thread(r.index_tools)
        retriever = r
        print("[INFO] Background: Retriever Ready.")
    except Exception as e:
        print(f"[ERROR] Background Retriever Init Failed: {e}")
    finally:
        retriever_loading = False

# --- App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Plugin Router
app.include_router(add_agent.router)

@app.on_event("startup")
async def startup_event():
    print("[INFO] Initializing ToolBench Retriever...")
    asyncio.create_task(init_retriever())


# --- Simulation State ---

class AgentState:
    """Encapsulates the state for a single agent (Baseline or Watermarked)"""
    def __init__(self, task_data: Dict, role: str):
        self.role = role # 'baseline' or 'watermarked'
        self.task = copy.deepcopy(task_data) # Deep copy to ensure independent modification
        
        # ToolBench Adapter State
        # For this demo, we use a simplified Adapter relying on LLM to propose JSON
        self.adapter = ToolBenchAdapter(TOOL_DATA_ROOT)
        self.episode = self.adapter.prepare_episode(self.task)
        
        # Execution History
        self.trajectory = [] # List of {role, message}
        self.swarm_history: List[Dict[str, Any]] = []
        self.step_count = 0
        self.last_observation = ""
        self.done = False

class Session:
    def __init__(self, session_id: str, api_key: str, task_data: Dict, payload: str = "1101"):
        self.session_id = session_id
        self.start_time = time.time()
        
        # Common Config
        self.max_steps = 15
        
        # Agent States
        self.watermarked_state = AgentState(task_data, 'watermarked')
        self.baseline_state = AgentState(task_data, 'baseline')
        
        # Payload / Watermark State (Only for watermarked agent)
        self.bit_stream_str_raw = payload if payload else "1101" # Keep raw for reference
        # Initialize RLNC
        self.rlnc = DeterministicRLNC(self.bit_stream_str_raw)
        self.bit_index = 0
        
        # LLM Client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"
        self.evaluation_result = None # Store evaluation result

sessions: Dict[str, Session] = {}



def _extract_watermark(completion: Any) -> Optional[Dict[str, Any]]:
    try:
        extra = getattr(completion, "model_extra", None)
        if extra and isinstance(extra, dict) and extra.get("watermark"):
            return extra.get("watermark")
        extra = getattr(completion, "__pydantic_extra__", None)
        if extra and isinstance(extra, dict) and extra.get("watermark"):
            return extra.get("watermark")
        payload = completion.model_dump()
        return payload.get("watermark")
    except Exception:
        return None


def _extract_tool_calls(message: Any) -> List[Dict[str, Any]]:
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not raw_tool_calls:
        return []
    tool_calls: List[Dict[str, Any]] = []
    for call in raw_tool_calls:
        if hasattr(call, "model_dump"):
            tool_calls.append(call.model_dump())
        elif isinstance(call, dict):
            tool_calls.append(call)
        else:
            tool_calls.append(getattr(call, "__dict__", {}))
    return tool_calls


def _extract_tokens_used(completion: Any) -> float:
    tokens_used = 0.0
    try:
        usage = getattr(completion, "usage", None)
        if usage is not None:
            if hasattr(usage, "total_tokens"):
                tokens_used = float(getattr(usage, "total_tokens", 0) or 0)
            elif isinstance(usage, dict):
                tokens_used = float(usage.get("total_tokens", 0) or 0)
    except Exception:
        tokens_used = 0.0
    return tokens_used


def _build_baseline_step(
    completion: Any,
    latency: float,
    *,
    fallback_content: str = "",
    candidates: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        message = completion.choices[0].message if completion and completion.choices else None
    except Exception:
        message = None
    if message is None:
        return None
    content = (getattr(message, "content", None) or fallback_content or "").strip()
    tool_calls = _extract_tool_calls(message)
    action = ""
    action_name = ""
    tool_details = ""
    step_type = "other"
    final_answer = content or None
    if tool_calls:
        first = tool_calls[0]
        fn = first.get("function", {}) if isinstance(first.get("function"), dict) else {}
        name = fn.get("name") or first.get("name") or ""
        action_name = name or ""
        action = f"Call: {action_name}" if action_name else ""
        args = fn.get("arguments")
        if args is None:
            args = first.get("arguments")
        if isinstance(args, dict):
            tool_details = json.dumps(args, ensure_ascii=False)
        elif args is not None:
            tool_details = str(args)
        step_type = "tool"
        final_answer = None
    elif final_answer:
        step_type = "finish"
        action_name = "Finish"
        action = action_name

    distribution: List[Dict[str, Any]] = []
    parsed_payload: Dict[str, Any] = {}
    if content:
        try:
            parsed_payload = extract_json_payload(content)
        except Exception:
            parsed_payload = {}

    if candidates:
        ordered = list(dict.fromkeys(candidates))
        if action_name and action_name not in ordered:
            ordered.append(action_name)
        prob_output = content
        if action_name:
            use_fallback = False
            if prob_output:
                try:
                    payload = extract_json_payload(prob_output)
                    if not isinstance(payload, dict) or ("action_weights" not in payload and "action" not in payload):
                        use_fallback = True
                except Exception:
                    use_fallback = True
            else:
                use_fallback = True
            if use_fallback:
                prob_output = json.dumps({"action": action_name})
        prob_map = extract_and_normalize_probabilities(prob_output or "", ordered)
        distribution = [
            {
                "name": name,
                "prob": float(prob_map.get(name, 0.0)),
                "isSelected": name == action_name,
            }
            for name in ordered
        ]
        if not action_name and prob_map:
            action_name = max(prob_map.items(), key=lambda x: x[1])[0]

    if not action_name and isinstance(parsed_payload, dict):
        payload_action = parsed_payload.get("action") or parsed_payload.get("tool")
        if payload_action:
            action_name = str(payload_action)

    action_args = None
    if isinstance(parsed_payload, dict):
        raw_args = parsed_payload.get("action_args")
        if isinstance(raw_args, dict) and action_name:
            if action_name in raw_args:
                action_args = raw_args.get(action_name)
            else:
                action_args = raw_args
        elif raw_args is not None:
            action_args = raw_args

    if action_args is not None:
        try:
            tool_details = json.dumps(action_args, ensure_ascii=False)
        except Exception:
            tool_details = str(action_args)

    if action_name:
        if action_name == "Finish":
            step_type = "finish"
            if isinstance(action_args, dict) and action_args.get("final_answer"):
                final_answer = action_args.get("final_answer")
        else:
            step_type = "tool"
        action = f"Call: {action_name}" if action_name else action

    tokens_used = _extract_tokens_used(completion)
    if latency <= 0:
        latency = 0.001

    thought = _extract_thought_from_raw_output(content)
    if not thought:
        thought = "no thought"

    return {
        "thought": thought,
        "action": action,
        "toolDetails": tool_details,
        "distribution": distribution,
        "stepType": step_type,
        "finalAnswer": final_answer,
        "metrics": {"latency": float(latency), "tokens": float(tokens_used)},
    }


def _extract_thought_from_raw_output(raw_text: str) -> str:
    if not raw_text:
        return ""

    sanitized = raw_text
    for token in ('"prompt_trace"', '"scoring_messages"', '"execution_messages"'):
        cut = sanitized.find(token)
        if cut != -1:
            sanitized = sanitized[:cut]
            break

    try:
        payload = extract_json_payload(sanitized)
        if isinstance(payload, dict):
            thought_val = payload.get("thought")
            if isinstance(thought_val, str) and thought_val.strip():
                return thought_val.strip()
    except Exception:
        pass

    matches = re.findall(r'"thought"\s*:\s*"((?:\\.|[^"\\])*)"', sanitized, flags=re.DOTALL)
    if matches:
        candidate = matches[-1]
        try:
            return json.loads(f"\"{candidate}\"").strip()
        except Exception:
            return candidate.replace("\\n", "\n").strip()

    lowered = sanitized.lower()
    idx = lowered.rfind('"thought"')
    quote_char = '"'
    if idx == -1:
        idx = lowered.rfind("'thought'")
        quote_char = "'"
    if idx != -1:
        after = sanitized[idx + len(quote_char + "thought" + quote_char):]
        colon = after.find(":")
        if colon != -1:
            rest = after[colon + 1:].lstrip()
            if rest.startswith(("\"", "'")):
                q = rest[0]
                rest = rest[1:]
                buf = []
                escaped = False
                for ch in rest:
                    if escaped:
                        buf.append(ch)
                        escaped = False
                        continue
                    if ch == "\\":
                        buf.append(ch)
                        escaped = True
                        continue
                    if ch == q:
                        break
                    buf.append(ch)
                candidate = "".join(buf)
                try:
                    return json.loads(f"\"{candidate}\"").strip()
                except Exception:
                    return candidate.replace("\\n", "\n").strip()
            else:
                end = len(rest)
                for sep in (",", "\n", "\r", "}"):
                    pos = rest.find(sep)
                    if pos != -1:
                        end = min(end, pos)
                return rest[:end].strip()

    return ""


def _build_add_agent_step(
    watermark: Dict[str, Any],
    step_index: int,
    completion_content: Optional[str],
    completion_tool_calls: Optional[List[Dict[str, Any]]],
    *,
    latency: float,
    tokens: float,
) -> Dict[str, Any]:
    frontend = watermark.get("frontend_data") or {}
    diff = frontend.get("distribution_diff") or []
    distribution = []
    for item in diff:
        prob = item.get("original_prob")
        if prob is None:
            prob = item.get("watermarked_prob", 0)
        distribution.append(
            {
                "name": item.get("action") or "",
                "prob": float(prob),
                "isSelected": bool(item.get("is_selected")),
            }
        )
    action = watermark.get("action") or ""
    action_args = watermark.get("action_args") or {}
    raw_text = watermark.get("raw_llm_output") or ""
    thought = _extract_thought_from_raw_output(raw_text)
    if not thought:
        thought = "no thought"

    bits_embedded = frontend.get("watermark_meta", {}).get("bits_embedded") or 0
    matrix_rows = [[1] for _ in range(int(bits_embedded))]
    final_answer = (completion_content or "").strip()
    if not final_answer and raw_text:
        final_answer = raw_text.strip()
    has_tool_calls = bool(completion_tool_calls)
    step_type = "tool" if action else "other"
    if final_answer and not has_tool_calls:
        step_type = "finish"
    return {
        "stepIndex": step_index,
        "thought": thought,
        "action": f"Call: {action}" if action else "",
        "toolDetails": json.dumps(action_args, ensure_ascii=False),
        "distribution": distribution,
        "watermark": {
            "bits": watermark.get("decoded_bits") or "",
            "matrixRows": matrix_rows,
            "rankContribution": len(matrix_rows),
        },
        "stepType": step_type,
        "metrics": {"latency": float(latency), "tokens": float(tokens)},
        "finalAnswer": final_answer or None,
    }

class InitRequest(BaseModel):
    apiKey: Optional[str] = None
    scenarioId: str
    payload: Optional[str] = None

class CustomInitRequest(BaseModel):
    apiKey: Optional[str] = None
    query: str
    payload: Optional[str] = None

class StepRequest(BaseModel):
    sessionId: str

class ContinueRequest(BaseModel):
    sessionId: str
    prompt: str




# --- Helpers ---
def build_messages(query: str, tool_summaries: List[str], admissible_commands: List[str]) -> List[Dict]:
    # Construct System Prompt compatible with ToolBench
    sys_prompt = f"""You are an Auto-GPT agent. Result of your previous step is passed to you.
You have access to the following tools:
{json.dumps(tool_summaries, indent=2)}

You must respond in JSON format with 'thought', 'action', 'action_args', and 'action_weights'.
'action_weights' must be a JSON object mapping EVERY valid action to a STRICTLY POSITIVE number (> 0, not necessarily normalized; the server will normalize them to sum to 1).
IMPORTANT: Do NOT output zeros. Every valid action must have weight > 0 (use small values like 1e-3 if needed).
Valid actions are: {json.dumps(admissible_commands)}
If you have enough information, use "Finish" and provide the final answer in "action_args" as {{"final_answer": "your answer"}}.
"""
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Task: {query}\nBegin!"}
    ]

def extract_and_normalize_probabilities(output: str, candidates: List[str]) -> Dict[str, float]:
    # DeepSeek chat API doesn't expose logprobs in this demo.
    # We therefore:
    #   1) Prefer model-provided `action_weights` if present (normalized).
    #   2) Otherwise, fall back to a biased-but-non-degenerate distribution (multiple "steps"),
    #      to avoid the "one huge + many identical tiny" shape that collapses bins in the UI.

    if not candidates:
        return {}
    if len(candidates) == 1:
        return {candidates[0]: 1.0}

    def _parse_json_payload(text: str) -> Optional[Dict]:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    def _coerce_nonneg_float(value) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0
        if not (v == v) or v == float("inf") or v == float("-inf"):
            return 0.0
        return v if v > 0.0 else 0.0

    def _geometric_fallback(top_action: Optional[str]) -> Dict[str, float]:
        # Biased-but-non-degenerate distribution: top_action gets fixed mass,
        # the rest get a geometric decay so p2>p3>... (no flat plateau).
        ratio = 0.75

        if top_action is not None and top_action in candidates:
            top_mass = 0.4
            others = [c for c in candidates if c != top_action]
            denom = sum((ratio**i) for i in range(len(others)))
            if denom <= 0.0:
                return uniform_prob(candidates)

            remainder = 1.0 - top_mass
            scores = {top_action: top_mass}
            for i, c in enumerate(others):
                scores[c] = remainder * (ratio**i) / denom
            return scores

        denom = sum((ratio**i) for i in range(len(candidates)))
        if denom <= 0.0:
            return uniform_prob(candidates)
        return {c: (ratio**i) / denom for i, c in enumerate(candidates)}

    def _mix_distributions(a: Dict[str, float], b: Dict[str, float], alpha: float) -> Dict[str, float]:
        alpha = max(0.0, min(1.0, alpha))
        mixed = {c: alpha * float(a.get(c, 0.0)) + (1.0 - alpha) * float(b.get(c, 0.0)) for c in candidates}
        total = sum(mixed.values())
        if total <= 0.0:
            return uniform_prob(candidates)
        return {k: v / total for k, v in mixed.items()}

    data = _parse_json_payload(output) or {}
    chosen = data.get("action", "Finish")

    raw_weights = data.get("action_weights", None)
    if raw_weights is not None:
        weights: Dict[str, float] = {}
        valid = True

        if isinstance(raw_weights, dict):
            for c in candidates:
                if c not in raw_weights:
                    valid = False
                    break
            for c in candidates:
                weights[c] = _coerce_nonneg_float(raw_weights.get(c, 0.0))
                if weights[c] <= 0.0:
                    valid = False
                    break
        elif isinstance(raw_weights, list) and len(raw_weights) == len(candidates):
            for i, c in enumerate(candidates):
                weights[c] = _coerce_nonneg_float(raw_weights[i])
                if weights[c] <= 0.0:
                    valid = False
                    break
        else:
            valid = False

        if valid:
            total = sum(weights.values())
            if total > 0.0:
                normalized = {k: v / total for k, v in weights.items()}
                top_action = max(normalized.items(), key=lambda x: x[1])[0]
                max_prob = normalized.get(top_action, 0.0)
                if max_prob > 0.9:
                    return _mix_distributions(normalized, _geometric_fallback(top_action), 0.6)
                return normalized

    return _geometric_fallback(chosen if chosen in candidates else None)


def _parse_action_args_from_output(model_output: str, chosen: str) -> Dict[str, Any]:
    try:
        start = model_output.find("{")
        end = model_output.rfind("}")
        json_str = model_output[start:end + 1] if start != -1 and end != -1 else "{}"
        data = json.loads(json_str)
        if "action_args" in data:
            raw_args = data["action_args"]
            if isinstance(raw_args, dict) and chosen in raw_args:
                return raw_args[chosen]
            if isinstance(raw_args, dict):
                return raw_args
        return {}
    except Exception:
        return {}

@app.post("/api/init")
async def init_session(req: InitRequest):
    session_id = f"sess_{int(time.time())}_{req.scenarioId}"
    
    # Load Scenario Data
    # In a real app, this would load from disk/db. 
    # We will use the 'retriever' to find tools for the query if scenario query is custom?
    # For fixed scenarios, we might already have the tool list. 
    # Let's assume we retrieve dynamically for "Live" demo always, or use cached.
    
    task = {
        "query": "Solve task " + req.scenarioId, 
        "api_list": [], # Will be empty, adapter handles fallback or we retrieve
        "id": req.scenarioId,
        "payload_str": req.payload 
    }
    
    # Try retrieve real tools if we know the query? 
    # For now, start empty or basic.
    
    api_key = _resolve_api_key(req.apiKey)
    session = Session(session_id, api_key, task, req.payload)
    sessions[session_id] = session
    
    print(f"[INFO] Session {session_id} initialized with Payload: '{task['payload_str']}'")
    
    return {
        "sessionId": session_id,
        "task": {
            "query": task.get("query"),
            "id": req.scenarioId
        },
        "totalSteps": 0, # Start
        "payloadLength": len(req.payload) if req.payload else 16
    }

@app.post("/api/init_custom")
async def init_custom_session(req: CustomInitRequest):
    session_id = f"sess_{int(time.time())}_custom"
    
    print(f"\n[INFO] >>> RECEIVED CUSTOM PROMPT: '{req.query}' <<<")
    
    api_list = []
    if retriever_loading:
        print("[WARN] Retriever is still loading...")
    elif retriever:
        api_list = retriever.retrieve(req.query, top_k=5)
    
    if not api_list:
        print("[WARN] Retriever found no tools or retrieval failed (or loading).")
        
    task = {
        "query": req.query,
        "api_list": api_list,
        "id": "custom_generated",
        "payload_str": req.payload
    }
    
    api_key = _resolve_api_key(req.apiKey)
    session = Session(session_id, api_key, task, req.payload)
    sessions[session_id] = session
    
    return {
        "sessionId": session_id,
        "task": {
            "query": task.get("query"),
            "id": task.get("id"),
            "retrieved_tools_count": len(api_list)
        },
        "totalSteps": 0,
        "payloadLength": len(task.get("payload_str", req.payload)) if req.payload else 16 
    }




@app.delete("/api/scenarios/clear_all")
async def clear_all_history():
    """Clear all conversation history from database"""
    try:
        deleted_count = db.clear_all_conversations()
        print(f"[INFO] Cleared all history: {deleted_count} conversations deleted")
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        print(f"[ERROR] Clear all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scenarios/batch_delete")
async def batch_delete_scenarios(req: dict):
    """Batch delete multiple scenarios"""
    try:
        scenario_ids = req.get("ids", [])
        if not scenario_ids:
            return {"status": "success", "deleted_count": 0}
        
        deleted_count = 0
        for scenario_id in scenario_ids:
            if db.delete_conversation(scenario_id):
                deleted_count += 1
        
        print(f"[INFO] Batch deleted {deleted_count} scenarios")
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        print(f"[ERROR] Batch delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scenarios/{scenario_id}/toggle_pin")
async def toggle_pin_scenario(scenario_id: str):
    """Toggle pin status of a conversation"""
    try:
        success = db.toggle_pin(scenario_id)
        if success:
            print(f"[INFO] Toggled pin status for scenario {scenario_id}")
            return {"status": "success", "id": scenario_id}
        else:
            raise HTTPException(status_code=404, detail="Scenario not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Toggle pin failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/scenarios/{scenario_id}")
async def delete_scenario(scenario_id: str):
    """Delete a conversation from database"""
    try:
        deleted = db.delete_conversation(scenario_id)
        if deleted:
            print(f"[INFO] Deleted scenario {scenario_id} from database")
            return {"status": "success", "id": scenario_id}
        else:
            raise HTTPException(status_code=404, detail="Scenario not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

class GenerateTitleRequest(BaseModel):
    history: List[Dict] # List of {role, content/message}

@app.post("/api/generate_title")
async def generate_title(req: GenerateTitleRequest):
    try:
        # Extract user messages to summarize
        # Limit to first few turns for title generation
        conversation_text = ""
        for turn in req.history[:6]:
            role = turn.get("role", "")
            content = turn.get("message") or turn.get("content") or ""
            if role == "user":
                conversation_text += f"User: {content}\n"
            elif role == "assistant":
                conversation_text += f"Assistant: {content}\n"
        
        if not conversation_text:
            return {"title": "New Conversation"}

        # Use a quick call to generate title
        # We can use the same client
        # Create a temporary client if needed, or reuse one from a session if available?
        # We don't have a session ID here necessarily.
        # But we initialized 'sessions' with keys. We can just instantiate a generic client.
        # However, to avoid global client init if not needed, we can just pick one active session or init a temporary one.
        # OR better: init a global client for utility tasks.
        
        # NOTE: In this demo, we assume we have an API Key.
        # But this request comes from frontend. Does it have API Key?
        # The frontend might not pass API key here if it's "auto save".
        # We should ideally pass API Key in request or reuse global environment variable.
        # For this demo, let's assume we reuse a valid API key from any active session or environment.
        # If no active session, we might fail.
        # Let's check if we have any active session to steal credentials or use a default if configured.
        
        api_key = None
        if sessions:
            api_key = list(sessions.values())[0].client.api_key
        
        if not api_key:
             return {"title": "New Conversation (Untitled)"}

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Generate a very concise title (3-6 words) for this conversation. Output ONLY the title, no quotes."},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.7,
            max_tokens=20
        )
        
        title = response.choices[0].message.content.strip().replace('"', '')
        return {"title": title}

    except Exception as e:
        print(f"[ERROR] Title generation failed: {e}")
        return {"title": "New Conversation"}



def uniform_prob(commands: List[str]) -> Dict[str, float]:
    p = 1.0 / len(commands) if commands else 0
    return {c: p for c in commands}

class RestoreSessionRequest(BaseModel):
    apiKey: Optional[str] = None
    scenarioId: str

@app.post("/api/restore_session")
async def restore_session(req: RestoreSessionRequest):
    """Restore session from database"""
    print(f"[INFO] Restore session request for scenarioId: {req.scenarioId}")
    
    # 1. Load saved scenario from database
    data = db.get_conversation(req.scenarioId)
    
    if not data:
        print(f"[ERROR] Scenario {req.scenarioId} not found in database")
        # List all available conversations for debugging
        all_convs = db.list_conversations(limit=10)
        print(f"[INFO] Available conversations: {[c['id'] for c in all_convs]}")
        raise HTTPException(status_code=404, detail="Saved scenario not found")
    
    print(f"[INFO] Found scenario in database: {data.get('id')}, steps: {len(data.get('steps', []))}")

    # 2. Init Session
    session_id = f"sess_{int(time.time())}_{req.scenarioId}_restored"
    
    # Extract task details from saved data
    # Saved data has 'steps', 'userQuery', etc.
    task = {
        "query": data.get("userQuery") or "Restored Task",
        "api_list": [], # We will rely on retrieval for next steps or assume stateless
        "id": req.scenarioId,
        "payload_str": data.get("payload") or "11001101"
    }
    
    # 3. Create Session
    api_key = _resolve_api_key(req.apiKey)
    session = Session(session_id, api_key, task, task["payload_str"])
    
    # 4. Reconstruct Trajectory from Steps
    # This is "best effort" mapping from UI-steps to internal-trajectory
    # UI Step Types: 'user_input', 'thought' (with action/tool), 'tool', 'finish'
    
    watermarked_trajectory = []
    baseline_trajectory = []
    
    steps = data.get("steps", [])
    
    # We need to map steps to (User, Assistant, Tool) messages.
    # Logic:
    # - if stepType == 'user_input': -> User Message
    # - if stepType == 'thought' or 'finish': -> Assistant Message (reconstruct JSON)
    # - if stepType == 'tool': -> Tool Message (Observation)
    
    # Let's iterate and reconstruct
    for step in steps:
        s_type = step.get("stepType", "thought")
        
        if s_type == "user_input":
            # User messages are the same for both agents
            user_msg = {"role": "user", "message": step.get("thought") or step.get("action")}
            watermarked_trajectory.append(user_msg)
            baseline_trajectory.append(user_msg)
            
        elif s_type in ["thought", "finish", "tool"]:
            # Reconstruct Watermarked Agent's messages
            thought = step.get("thought", "")
            action = step.get("action", "")
            final_answer = step.get("finalAnswer")
            
            # Helper to parse "Call: ToolName" -> ToolName
            chosen_tool = "Finish"
            if action.startswith("Call: "):
                chosen_tool = action.replace("Call: ", "").strip()
            elif action == "Finish":
                chosen_tool = "Finish"
            
            # Reconstruct Dict for watermarked agent
            model_out_dict = {
                "action": chosen_tool,
                "action_args": {},
                "thought": thought
            }
            
            if chosen_tool == "Finish" and final_answer:
                 model_out_dict["action_args"] = { "final_answer": final_answer }
            
            # Store as string (mocking the LLM raw output)
            watermarked_trajectory.append({"role": "assistant", "message": json.dumps(model_out_dict)})
            
            # Add observation for watermarked agent
            obs = step.get("toolDetails") or step.get("observation")
            if obs and chosen_tool != "Finish":
                watermarked_trajectory.append({"role": "tool", "message": obs})
            
            # Reconstruct Baseline Agent's messages (if exists)
            baseline_data = step.get("baseline")
            if baseline_data:
                baseline_thought = baseline_data.get("thought", "")
                baseline_action = baseline_data.get("action", "")
                baseline_final_answer = baseline_data.get("finalAnswer")
                
                # Parse baseline action
                baseline_tool = "Finish"
                if baseline_action.startswith("Call: "):
                    baseline_tool = baseline_action.replace("Call: ", "").strip()
                elif baseline_action == "Finish":
                    baseline_tool = "Finish"
                
                # Reconstruct Dict for baseline agent
                baseline_model_dict = {
                    "action": baseline_tool,
                    "action_args": {},
                    "thought": baseline_thought
                }
                
                if baseline_tool == "Finish" and baseline_final_answer:
                    baseline_model_dict["action_args"] = { "final_answer": baseline_final_answer }
                
                baseline_trajectory.append({"role": "assistant", "message": json.dumps(baseline_model_dict)})
                
                # Add observation for baseline agent
                baseline_obs = baseline_data.get("toolDetails") or baseline_data.get("observation")
                if baseline_obs and baseline_tool != "Finish":
                    baseline_trajectory.append({"role": "tool", "message": baseline_obs})
            else:
                # If no baseline data, copy watermarked data
                baseline_trajectory.append({"role": "assistant", "message": json.dumps(model_out_dict)})
                if obs and chosen_tool != "Finish":
                    baseline_trajectory.append({"role": "tool", "message": obs})

    # Hydrate both agents with their respective trajectories
    session.watermarked_state.trajectory = watermarked_trajectory
    session.baseline_state.trajectory = baseline_trajectory
    
    # Set step count
    session.watermarked_state.step_count = len(steps)
    session.baseline_state.step_count = len(steps)
    
    # Store session
    sessions[session_id] = session
    
    print(f"[INFO] Restored session {session_id} with {len(watermarked_trajectory)} watermarked turns and {len(baseline_trajectory)} baseline turns.")
    print(f"[INFO] Session stored in sessions dict. Total sessions: {len(sessions)}")
    print(f"[INFO] Session keys: {list(sessions.keys())}")
    
    return {
        "sessionId": session_id,
        "task": {
             "query": task["query"],
             "id": req.scenarioId
        },
        "restoredSteps": len(steps)
    }

@app.post("/api/continue")
async def continue_session(req: ContinueRequest):
    if req.sessionId not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sess = sessions[req.sessionId]
    
    # Retrieve new tools for the continuation prompt
    print(f"\n[INFO] >>> RECEIVED CONTINUE PROMPT: '{req.prompt}' <<<\n")
    if retriever:
        new_tools = retriever.retrieve(req.prompt, top_k=5)
        if new_tools:
            print(f"[INFO] Retrieved {len(new_tools)} new tools for continuation.")
            
            # Helper to update agent state with new tools
            def update_agent_tools(agent_state: AgentState):
                # Basic dedup check
                current_tools = agent_state.task.get("api_list", [])
                existing_names = {t.get("func_name") or t.get("api_name") for t in current_tools}
                
                for tool in new_tools:
                    t_name = tool.get("func_name") or tool.get("api_name")
                    if t_name not in existing_names:
                        current_tools.append(tool)
                        existing_names.add(t_name)
                
                agent_state.task["api_list"] = current_tools
                
                # CRITICAL: Re-initialize episode to refresh tool summaries and admissible commands
                try:
                    updated_episode = agent_state.adapter.prepare_episode(agent_state.task)
                    agent_state.episode["tool_summaries"] = updated_episode["tool_summaries"]
                    agent_state.episode["admissible_commands"] = updated_episode["admissible_commands"]
                except Exception as e:
                    print(f"[ERROR] Failed to refresh episode context for {agent_state.role}: {e}")

            # Update both agents
            update_agent_tools(sess.watermarked_state)
            update_agent_tools(sess.baseline_state)
            print("[INFO] Updated tools for both agents.")
            
    
    # Append user prompt to trajectory for both
    sess.watermarked_state.trajectory.append({"role": "user", "message": req.prompt})
    sess.baseline_state.trajectory.append({"role": "user", "message": req.prompt})
    
    # Extend max steps to allow continuation
    sess.max_steps += 10
    
    # CRITICAL: Reset done state so agents continue
    sess.watermarked_state.done = False
    sess.baseline_state.done = False
    
    return {"status": "success", "message": "Session continued", "new_max_steps": sess.max_steps}

@app.post("/api/step")
async def step_session(req: StepRequest):
    if req.sessionId not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sess = sessions[req.sessionId]
    use_shared_thought = (os.getenv("AGENTMARK_BASELINE_SHARE_THOUGHT") or "0").strip().lower() not in {"0", "false", "no"}
    use_swarm_baseline = (os.getenv("AGENTMARK_BASELINE_USE_SWARM") or "1").strip().lower() not in {"0", "false", "no"}
    # if use_shared_thought:
    #     use_swarm_baseline = False
    
    if sess.watermarked_state.step_count >= sess.max_steps:
        # Return immediate JSON for consistency if done
        def immediate_done():
            yield json.dumps({
                "type": "result",
                "data": {
                    "agent": "watermarked",
                    "done": True,
                    "thought": "Max steps reached",
                    "action": "Finish",
                    "observation": "",
                    "final_answer": "",
                    "distribution": [],
                    "metrics": {"latency": 0.0, "tokens": 0.0},
                },
            }) + "\n"
            yield json.dumps({
                "type": "result",
                "data": {
                    "agent": "baseline",
                    "done": True,
                    "thought": "Max steps reached",
                    "action": "Finish",
                    "observation": "",
                    "final_answer": "",
                    "distribution": [],
                    "metrics": {"latency": 0.0, "tokens": 0.0},
                },
            }) + "\n"
        return StreamingResponse(immediate_done(), media_type="application/x-ndjson")


    # --- Generic Single Agent Step Function ---
    async def step_single_agent(
        agent_state: AgentState,
        is_watermarked: bool,
        output_queue: asyncio.Queue,
        *,
        mirror_agent: Optional[str] = None,
        capture_output: bool = False,
    ):
        step_start_time = time.time()
        
        # Check done state
        if agent_state.done:
            final_data = {
                "agent": agent_state.role,
                "thought": "",
                "action": "Finish",
                "observation": "",
                "done": True,
                "final_answer": "",
                "distribution": [],
                "metrics": {
                    "latency": 0.0,
                    "tokens": 0.0,
                },
            }

            shared_payload = None
            if is_watermarked:
                watermark_data = {
                    "bits": "",
                    "matrixRows": [],
                    "rankContribution": 0,
                }
                return final_data, watermark_data, 0, shared_payload

            return final_data, None, 0, shared_payload


            try:
                # Swarm Native Execution (Threaded for Sync Client)
                from swarm import Swarm, Agent
                
                tool_summaries = agent_state.episode["tool_summaries"]
                tools_override = _build_toolbench_tools(tool_summaries)
                loop = asyncio.get_running_loop()
                
                # Use shared helper
                funcs = _create_dynamic_swarm_tools(
                    tool_summaries, 
                    agent_state.adapter, 
                    agent_state.task, 
                    output_queue, 
                    loop, 
                    agent_state.role
                )

                # Swarm Agent Definition
                swarm_agent = Agent(
                    name="Swarm Assistant",
                    instructions="You are a helpful agent. Use the provided tools to solve the task. Call a tool when needed; otherwise, respond with the final answer.",
                    functions=funcs 
                )
                
                # ... run_swarm_sync ...
                
                # 3. Wrapper for Thread Execution
                def run_swarm_sync():
                    # Initialize Swarm client (Sync)
                    client_sync = sess.client
                    swarm = Swarm(client=client_sync)
                    
                    # Convert history
                    if not agent_state.swarm_history:
                        agent_state.swarm_history.append({
                            "role": "user", 
                            "content": f"Task: {agent_state.task.get('query', '')}\nBegin!"
                        })
                    
                            
                    # Run and Stream
                    print(f"[DEBUG] Starting Swarm run_and_stream with model {sess.model}")
                    stream = swarm.run_and_stream(
                        agent=swarm_agent,
                        messages=agent_state.swarm_history,
                        model_override=sess.model,
                        tools_override=tools_override,
                        execute_tools=True,
                        debug=True
                    )
                    
                    # Consume stream
                    response_obj = None
                    full_content = ""
                    
                    for chunk in stream:
                        print(f"[DEBUG] Swarm Chunk: {str(chunk)[:100]}") # Log first 100 chars
                        if "response" in chunk:
                            response_obj = chunk["response"]
                            break
                        
                        if "content" in chunk and chunk["content"]:
                            c = chunk["content"]
                            full_content += c
                            # Stream thought to UI
                            loop.call_soon_threadsafe(
                                output_queue.put_nowait,
                                {
                                    "type": "thought",
                                    "agent": agent_state.role,
                                    "content": c
                                }
                            )

                    
                    return response_obj

                # 4. Await Thread
                response = await asyncio.to_thread(run_swarm_sync)
                
                # 5. Process Final Result
                if response:
                    new_messages = response.messages
                    agent_state.swarm_history.extend(new_messages)
                    
                    # Convert new_messages to our trajectory format
                    final_answer_text = ""
                    done = False
                    
                    for msg in new_messages:
                        role = msg.get("role")
                        content = msg.get("content") or ""
                        
                        if role == "assistant":
                            tool_calls = msg.get("tool_calls")
                            
                            if tool_calls:
                                for tc in tool_calls:
                                    fn = tc.get("function", {})
                                    t_name = fn.get("name")
                                    t_args = fn.get("arguments")
                                    
                                    # Append to trajectory
                                    ui_msg = {
                                        "thought": content,
                                        "action": t_name,
                                        "action_args": json.loads(t_args) if t_args else {},
                                        "action_weights": {} 
                                    }
                                    agent_state.trajectory.append({"role": "assistant", "message": json.dumps(ui_msg)})
                                    
                                    # STREAM INTERMEDIATE STEP COMPLETION
                                    step_data = {
                                        "agent": agent_state.role,
                                        "thought": content or "Calling tool...",
                                        "action": f"Call: {t_name}",
                                        "observation": "", # Tool output comes in next msg role='tool'
                                        "done": False,
                                        "final_answer": "",
                                        "distribution": [],
                                        "stepIndex": agent_state.step_count, # Current index
                                        "metrics": {"latency": 0.1, "tokens": 0.0}
                                    }
                                    loop.call_soon_threadsafe(
                                        output_queue.put_nowait,
                                        {
                                            "type": "result", # Treat as a result chunk
                                            "data": step_data
                                        }
                                    )
                                    
                                    agent_state.step_count += 1
                                    
                            else:
                                # Final Answer / Plain content
                                ui_msg = {
                                    "thought": content,
                                    "action": "Finish",
                                    "action_args": {"final_answer": content},
                                    "action_weights": {}
                                }
                                agent_state.trajectory.append({"role": "assistant", "message": json.dumps(ui_msg)})
                                final_answer_text = content
                                done = True
                                agent_state.step_count += 1

                        elif role == "tool":
                            # Tool Output
                            agent_state.trajectory.append({"role": "tool", "message": content})
                            agent_state.last_observation = content
                
                    agent_state.done = done

                    step_latency = time.time() - step_start_time
                    
                    final_data = {
                        "agent": agent_state.role,
                        "thought": final_answer_text or "Finished turn.",
                        "action": "Finish" if done else "Yield", # Swarm yields back control?
                        "observation": agent_state.last_observation,
                        "done": done,
                        "final_answer": final_answer_text,
                        "distribution": [], # NO BLUE BARS
                        "stepIndex": agent_state.step_count - 1,
                        "metrics": {"latency": step_latency, "tokens": 0.0},
                    }
                    return final_data, None, 0, None
                
                else:
                    raise Exception("Swarm returned no response")

            except Exception as e:
                print(f"[ERROR] Swarm baseline error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to avoid hanging
                agent_state.done = True
                return {
                    "agent": agent_state.role,
                    "thought": f"Error: {str(e)}",
                    "action": "Finish",
                    "done": True,
                    "metrics": {"latency": 0, "tokens": 0}
                }, None, 0, None

        # 1. Build Messages
        messages = build_messages(
            query=agent_state.task.get("query", ""),
            tool_summaries=agent_state.episode["tool_summaries"],
            admissible_commands=agent_state.episode["admissible_commands"]
        )
        # Add history
        for turn in agent_state.trajectory:
            if turn["role"] == "assistant":
                 messages.append({"role": "assistant", "content": turn["message"]})
            elif turn["role"] == "tool":
                 # Simulate tool output as user message (ToolBench style)
                 messages.append({"role": "user", "content": f"Observation:\n{turn['message']}\nContinue Thought/Action/Action Input."})
            elif turn["role"] == "user":
                 messages.append({"role": "user", "content": turn["message"]})

        # 2. Call LLM
        model_output = ""
        try:
            print(f"[DEBUG] Call LLM ({agent_state.role})")
            # ENABLE STREAMING
            response_stream = await sess.async_client.chat.completions.create(
                model=sess.model, 
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                stream=True 
            )
            
            async for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    model_output += content
                    # Stream thought chunk
                    # We assume everything is thought until we parse it fully later, or we just stream raw text.
                    # Ideally we would only stream if it's NOT inside a JSON structure, but for "Thinking...", 
                    # showing the raw generation is fine and "hacky" but works for visual effect.
                    await output_queue.put({
                        "type": "thought",
                        "agent": agent_state.role,
                        "content": content
                    })
                    if mirror_agent:
                        await output_queue.put({
                            "type": "thought",
                            "agent": mirror_agent,
                            "content": content
                        })

        except Exception as e:
            print(f"[ERROR] Step Model Error ({agent_state.role}): {e}")
            model_output = json.dumps({
                "action_weights": uniform_prob(agent_state.episode["admissible_commands"]),
                "action_args": {},
                "thought": "API Call Failed. Using fallback."
            })

        # 3. Process Result
        probs = extract_and_normalize_probabilities(model_output, agent_state.episode["admissible_commands"])
        if not probs:
            probs = uniform_prob(agent_state.episode["admissible_commands"])
            
        effective_probs = probs.copy() 
        
        chosen = "Finish"
        consumed_bits = 0
        
        # Sampling Strategy
        if is_watermarked:
            # Differential Sampling
            bit_before = sess.bit_index
            
            # 1. Fetch chunk from RLNC
            # We need enough bits for the sampler. The sampler typically consumes log2(N) bits, plus potentially more.
            # Let's fetch a safe chunk of 64 bits from the infinite stream
            # The sampler takes specific # of bits.
            # Ideally the sampler should take the stream object or we guess.
            # Our `sample_behavior_differential` implementation takes `bit_stream` as string.
            # We generate a chunk of 64 bits starting at bit_index.
            
            chunk_length = 64
            rlnc_chunk = sess.rlnc.get_stream(start_index=sess.bit_index, length=chunk_length)
            
            # 2. Call Real Sampler
            # Note: The real sampler signature is:
            # sample_behavior_differential(probabilities, bit_stream, bit_index, context_for_key=None, history_responses=None, seed=None, round_num=0)
            # IMPORTANT: The `bit_index` arg in sampler acts as an offset into the passed `bit_stream`.
            # Since we pass a fresh chunk, we should pass index 0 to the sampler, OR pass the full virtual stream?
            # Passing full virtual stream is impossible.
            # We pass the chunk, and tell sampler index is 0 relative to chunk.
            
            chosen, target_list, consumed_bits, context_used = sample_behavior_differential(
                probabilities=effective_probs,
                bit_stream=rlnc_chunk,
                bit_index=0, # relative to chunk
                context_for_key=agent_state.last_observation, # context
                round_num=agent_state.step_count
            )
            
            # consumed_bits is how many bits from chunk were used.
            # sess.bit_index += consumed_bits (Done below)

        else:
            # Baseline: Max Prob (Greedy) or simple Random
            # Greedy for stability
            if effective_probs:
                chosen = max(effective_probs.items(), key=lambda x: x[1])[0]
            else:
                 chosen = "Finish"

        # Parse Action Args
        action_args = _parse_action_args_from_output(model_output, chosen)

        action_obj = {"tool": chosen, "arguments": action_args}
        
        # Update History
        agent_state.trajectory.append({"role": "assistant", "message": model_output})
        
        # Thought Extraction
        thought = ""
        if "Thought:" in model_output:
            thought = model_output.split("Thought:")[1].split("{")[0].strip()
        elif "thought:" in model_output:
             thought = model_output.split("thought:")[1].split("{")[0].strip()
        else:
            parts = model_output.split("{", 1)
            if len(parts) > 1 and parts[0].strip():
                thought = parts[0].strip().replace("```json", "").replace("```", "").strip()
            else:
                try:
                    start = model_output.find("{")
                    end = model_output.rfind("}")
                    if start != -1 and end != -1:
                       json_str = model_output[start:end+1]
                       data = json.loads(json_str)
                       thought = data.get("thought", "")
                except:
                    pass
        
        # Fallback: if thought is still empty but we have content
        if not thought and model_output.strip() and not model_output.strip().startswith("{"):
             # Just take the text before the first brace
             thought = model_output.split("{")[0].strip()

        shared_payload = None
        if capture_output:
            shared_payload = {
                "model_output": model_output,
                "probs": probs,
                "thought": thought,
            }

        # Extract Final Answer
        final_answer_text = ""
        if chosen == "Finish":
            print(f"\n[DEBUG] RAW MODEL OUTPUT (Finish): {model_output}\n")
            if isinstance(action_args, dict):
                final_answer_text = action_args.get("final_answer", "")
            elif isinstance(action_args, str):
                final_answer_text = action_args
            
            # Fallback text extraction for final answer if JSON failed
            if not final_answer_text:
                import re
                try:
                    # Try to find "Final Answer:" pattern
                    fa_match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE | re.DOTALL)
                    if fa_match:
                         final_answer_text = fa_match.group(1).strip()
                except:
                    pass
            
            # If thought is empty but we have final answer, use final answer as thought for display if needed
            # if not thought and final_answer_text:
            #     thought = "Task Completed." 

        # Execute Tool
        step_result_obs = await asyncio.to_thread(agent_state.adapter.step, action_obj, agent_state.episode["tool_summaries"], state=agent_state.task)
        observation = step_result_obs["observation"]
        done = step_result_obs.get("done", False) or chosen == "Finish"
        
        agent_state.trajectory.append({"role": "tool", "message": observation})
        agent_state.last_observation = observation
        agent_state.step_count += 1
        agent_state.done = done

        # Prepare Result Dict
        obs_display = observation
        if not done and len(observation) > 200:
            obs_display = observation[:200] + "..."

        step_latency = time.time() - step_start_time
        est_tokens = len(model_output) / 4 
        
        # Decide default thought
        default_thought = "Thinking..." if not done else ""
        
        final_data = {
            "agent": agent_state.role, # 'watermarked' or 'baseline'
            "thought": thought or default_thought,
            "action": f"Call: {chosen}",
            "observation": obs_display,
            "done": done,
            "final_answer": final_answer_text, # Explicitly send final answer
            "distribution": [{"name": k, "prob": v, "isSelected": k==chosen} for k, v in probs.items()],
            "stepIndex": agent_state.step_count - 1,
            "metrics": {
                "latency": step_latency,
                "tokens": est_tokens
            }
        }
        
        watermark_data = {}
        if is_watermarked:
            # Watermark Trace
            # Get the exact bits consumed
            embedded_bits = sess.rlnc.get_stream(start_index=sess.bit_index, length=consumed_bits)
            
            matrix_rows = []
            
            # Generate real RLNC coefficients for the consumed bits
            # The bits consumed were at absolute indices [sess.bit_index ... sess.bit_index + consumed_bits - 1]
            for i in range(consumed_bits):
                abs_idx = sess.bit_index + i
                coeffs = sess.rlnc._generate_coeffs(abs_idx)
                matrix_rows.append(coeffs)
            
            watermark_data = {
                "bits": embedded_bits,
                "matrixRows": matrix_rows,
                "rankContribution": len(embedded_bits)
            }
            return final_data, watermark_data, consumed_bits, shared_payload
        
        return final_data, None, 0, shared_payload


    async def step_baseline_from_shared(agent_state: AgentState, shared_payload: Dict[str, Any]):
        step_start_time = time.time()

        if agent_state.done:
            final_data = {
                "agent": agent_state.role,
                "thought": "",
                "action": "Finish",
                "observation": "",
                "done": True,
                "final_answer": "",
                "distribution": [],
                "metrics": {
                    "latency": 0.0,
                    "tokens": 0.0,
                },
            }
            return final_data, None, 0, None

        model_output = shared_payload.get("model_output") or ""
        probs = shared_payload.get("probs") or {}
        thought = shared_payload.get("thought") or ""

        admissible = agent_state.episode["admissible_commands"]
        effective_probs = probs if probs else uniform_prob(admissible)

        chosen = "Finish"
        if effective_probs:
            chosen = max(effective_probs.items(), key=lambda x: x[1])[0]

        if chosen not in admissible:
            chosen = "Finish"

        action_args = _parse_action_args_from_output(model_output, chosen)
        if chosen == "Finish" and not action_args and thought:
            action_args = {"final_answer": thought}

        action_obj = {"tool": chosen, "arguments": action_args}

        agent_state.trajectory.append({"role": "assistant", "message": model_output})

        step_result_obs = await asyncio.to_thread(
            agent_state.adapter.step,
            action_obj,
            agent_state.episode["tool_summaries"],
            state=agent_state.task,
        )
        observation = step_result_obs["observation"]
        done = step_result_obs.get("done", False) or chosen == "Finish"

        agent_state.trajectory.append({"role": "tool", "message": observation})
        agent_state.last_observation = observation
        agent_state.step_count += 1
        agent_state.done = done

        obs_display = observation
        if not done and len(observation) > 200:
            obs_display = observation[:200] + "..."

        step_latency = time.time() - step_start_time
        est_tokens = len(model_output) / 4 if model_output else 0.0

        final_answer_text = ""
        if chosen == "Finish":
            if isinstance(action_args, dict):
                final_answer_text = action_args.get("final_answer", "")
            if not final_answer_text:
                final_answer_text = thought

        distribution = [
            {"name": k, "prob": v, "isSelected": k == chosen}
            for k, v in effective_probs.items()
        ]

        final_data = {
            "agent": agent_state.role,
            "thought": thought or ("Thinking..." if not done else ""),
            "action": f"Call: {chosen}",
            "observation": obs_display,
            "done": done,
            "final_answer": final_answer_text if done else "",
            "distribution": distribution,
            "stepIndex": agent_state.step_count - 1,
            "metrics": {"latency": step_latency, "tokens": est_tokens},
        }

        return final_data, None, 0, None


    # MAIN GENERATOR
    async def step_generator():
        queue = asyncio.Queue()

        share_now = use_shared_thought and not sess.watermarked_state.done and not sess.baseline_state.done

        if share_now:
            task_wm = asyncio.create_task(
                step_single_agent(
                    sess.watermarked_state,
                    True,
                    queue,
                    mirror_agent="baseline",
                    capture_output=True,
                )
            )

            pending_tasks = {task_wm}

            while pending_tasks:
                queue_task = asyncio.create_task(queue.get())

                done, pending = await asyncio.wait(
                    pending_tasks | {queue_task},
                    return_when=asyncio.FIRST_COMPLETED
                )

                if queue_task in done:
                    chunk = queue_task.result()
                    yield json.dumps(chunk) + "\n"
                else:
                    queue_task.cancel()

                for t in done:
                    if t in pending_tasks:
                        pending_tasks.remove(t)

            while not queue.empty():
                chunk = await queue.get()
                yield json.dumps(chunk) + "\n"

            try:
                result_wm = await task_wm
                if result_wm:
                    final_data_wm, wm_trace, consumed, shared_payload = result_wm
                    sess.bit_index += consumed
                    final_data_wm["watermark"] = wm_trace
                    yield json.dumps({"type": "result", "data": final_data_wm}) + "\n"

                    if shared_payload:
                        result_bl = await step_baseline_from_shared(sess.baseline_state, shared_payload)
                        if result_bl:
                            final_data_bl, _, _, _ = result_bl
                            final_data_bl["watermark"] = { "bits": "", "matrixRows": [], "rankContribution": 0 }
                            yield json.dumps({"type": "result", "data": final_data_bl}) + "\n"

            except Exception as e:
                print(f"[ERROR] Task execution failed: {e}")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            return

        # Create tasks
        task_wm = asyncio.create_task(step_single_agent(sess.watermarked_state, True, queue))
        task_bl = asyncio.create_task(step_single_agent(sess.baseline_state, False, queue))

        pending_tasks = {task_wm, task_bl}

        while pending_tasks:
            # Wait for either queue item or task completion
            # We create a task for queue.get()
            queue_task = asyncio.create_task(queue.get())

            done, pending = await asyncio.wait(
                pending_tasks | {queue_task},
                return_when=asyncio.FIRST_COMPLETED
            )

            # Handle queue item
            if queue_task in done:
                chunk = queue_task.result()
                yield json.dumps(chunk) + "\n"
            else:
                queue_task.cancel()

            # Handle agent tasks completion
            for t in done:
                if t in pending_tasks:
                    pending_tasks.remove(t)

        # Consume any remaining queue items
        while not queue.empty():
            chunk = await queue.get()
            yield json.dumps(chunk) + "\n"

        # Get results
        try:
            result_wm = await task_wm
            result_bl = await task_bl

            # Unpack Watermark Result
            if result_wm:
                final_data_wm, wm_trace, consumed, _ = result_wm
                sess.bit_index += consumed
                final_data_wm["watermark"] = wm_trace
                yield json.dumps({"type": "result", "data": final_data_wm}) + "\n"

            # Unpack Baseline Result
            if result_bl:
                final_data_bl, _, _, _ = result_bl
                final_data_bl["watermark"] = { "bits": "", "matrixRows": [], "rankContribution": 0 }
                yield json.dumps({"type": "result", "data": final_data_bl}) + "\n"

        except Exception as e:
            print(f"[ERROR] Task execution failed: {e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(step_generator(), media_type="application/x-ndjson")
    
class EvaluateRequest(BaseModel):
    sessionId: str
    language: Optional[str] = "en"  # "en" or "zh"

@app.post("/api/evaluate")
async def evaluate_session(req: EvaluateRequest):
    print(f"[INFO] Evaluate request for session: {req.sessionId}")
    print(f"[INFO] Available sessions: {list(sessions.keys())}")
    
    if req.sessionId not in sessions:
        print(f"[ERROR] Session {req.sessionId} not found in sessions dict")
        raise HTTPException(status_code=404, detail="Session not found")
    
    sess = sessions[req.sessionId]
    
    # Language Instruction
    lang_instruction = "Reasoning must be in English."
    if req.language == "zh":
        lang_instruction = "请使用中文进行简要评价 (Reasoning must be in Chinese)."

    # helper to summarize trajectory
    def summarize_trajectory(traj):
        summary = ""
        for t in traj:
            role = t["role"]
            msg = t["message"]
            if role == "user":
                summary += f"User: {msg}\n"
            elif role == "assistant":
                # Try parse thought/action
                try:
                    data = json.loads(msg)
                    summary += f"Assistant Thought: {data.get('thought')}\nAssistant Action: {data.get('action')}\n"
                    if "final_answer" in data.get("action_args", {}):
                         summary += f"Assistant Final Answer: {data['action_args']['final_answer']}\n"
                except:
                    summary += f"Assistant: {msg}\n"
            elif role == "tool":
                summary += f"Tool Output: {msg[:200]}...\n"
        return summary

    baseline_summary = summarize_trajectory(sess.baseline_state.trajectory)
    watermarked_summary = summarize_trajectory(sess.watermarked_state.trajectory)
    query = sess.watermarked_state.task.get("query", "Unknown Task")

    # Anti-Bias: Randomize Order
    import random
    is_baseline_A = random.choice([True, False])

    if is_baseline_A:
        summary_A = baseline_summary
        summary_B = watermarked_summary
    else:
        summary_A = watermarked_summary
        summary_B = baseline_summary
    
    prompt = f"""Task: {query}
    
    Model A Trajectory/Answer:
    {summary_A}

    Model B Trajectory/Answer:
    {summary_B}

    Please evaluate Model A and Model B based on the task using criteria such as correctness, efficiency, and helpfulness.
    Provide a score (0-10) for each and a brief reason.
    {lang_instruction}
    
    You must output strictly in JSON format:
    {{
        "model_a_score": <float>,
        "model_b_score": <float>,
        "reason": "<string>"
    }}
    """
    
    try:
        completion = await sess.async_client.chat.completions.create(
            model=sess.model,
            messages=[
                {"role": "system", "content": "You are an impartial judge evaluating two AI models."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = completion.choices[0].message.content
        # parse json
        try:
            start = content.find("{")
            end = content.rfind("}")
            json_str = content[start:end+1]
            raw_result = json.loads(json_str)

            # Map back to specific models
            if is_baseline_A:
                result = {
                    "model_a_score": raw_result.get("model_a_score", 0),
                    "model_b_score": raw_result.get("model_b_score", 0),
                    "reason": raw_result.get("reason", "")
                }
            res_json = json.loads(json_str)

            swapped = not is_baseline_A
            # map scores back to Baseline vs Ours
            # If swapped (Model A was Watermarked), then A was Ours, B was Base.
            # We want output: model_a_score=Base, model_b_score=Ours
            final_result = {}
            if swapped:
                 final_result = {
                     "model_a_score": res_json.get("model_b_score", 0), # A (now Base) = B (was Base)
                     "model_b_score": res_json.get("model_a_score", 0), # B (now Ours) = A (was Ours)
                     "reason": res_json.get("reason", "")
                 }
            else:
                 final_result = res_json

            sess.evaluation_result = final_result # Persist in session

            # Update the database with evaluation result
            try:
                # Try to get the original scenario ID from the session
                original_id = sess.watermarked_state.task.get("id") or sess.baseline_state.task.get("id")
                
                # If session ID contains "_restored", extract the original ID
                if "_restored" in req.sessionId:
                    # Format: sess_timestamp_ORIGINAL_ID_restored
                    parts = req.sessionId.split("_")
                    if len(parts) >= 3:
                        # Find the part that's not "sess", not a timestamp, and not "restored"
                        for part in parts:
                            if part not in ["sess", "restored"] and not part.isdigit():
                                original_id = part
                                break
                
                print(f"[INFO] Attempting to save evaluation for scenario: {original_id}")
                
                if original_id:
                    existing = db.get_conversation(original_id)
                    if existing:
                        existing["evaluation"] = final_result
                        db.save_conversation(existing)
                        print(f"[INFO] Successfully updated evaluation for scenario {original_id} in database")
                    else:
                        print(f"[WARN] Scenario {original_id} not found in database")
                else:
                    print(f"[WARN] Could not determine original scenario ID from session {req.sessionId}")
            except Exception as save_err:
                print(f"[WARN] Failed to auto-save evaluation to database: {save_err}")
                import traceback
                traceback.print_exc()

            return final_result
            
        except Exception as e:
            print("JSON Parse Error:", e)
            return {"model_a_score": 0, "model_b_score": 0, "reason": "Failed to parse result"}
            
    except Exception as e:
        print("Evaluation Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
