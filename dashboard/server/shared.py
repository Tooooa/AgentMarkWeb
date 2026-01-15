
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from openai import OpenAI
import copy

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SWARM_ROOT = PROJECT_ROOT / "swarm"
TOOL_DATA_ROOT = PROJECT_ROOT / "experiments/toolbench/data/data/toolenv/tools"

def _load_root_dotenv() -> None:
    """Best-effort loader for PROJECT_ROOT/.env without external deps."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ.setdefault(key, value)
    except Exception:
        # Avoid breaking server startup due to .env formatting issues.
        return

_load_root_dotenv()

def _resolve_api_key(request_api_key: Optional[str]) -> str:
    candidate = (request_api_key or "").strip()
    if candidate:
        return candidate
    env_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if env_key:
        return env_key
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key
    raise HTTPException(
        status_code=400,
        detail="Missing API key. Provide apiKey in request or set DEEPSEEK_API_KEY (preferred) / OPENAI_API_KEY in environment.",
    )

def _get_proxy_base() -> str:
    return "http://127.0.0.1:8001/v1"

def _get_base_llm_base() -> str:
    return "https://api.deepseek.com"

def _build_proxy_client(api_key: str) -> OpenAI:
    """Builds an OpenAI client pointing to the AgentMark Proxy."""
    # The proxy is running locally, forwarding to the actual LLM.
    # The proxy expects the real API key to be passed (or handled internally if configured).
    # Here we pass the resolved key.
    return OpenAI(
        base_url=_get_proxy_base(),
        api_key=api_key
    )

def _create_dynamic_swarm_tools(tool_summaries: List[Dict], adapter: Any, task_state: Dict, queue: Optional[asyncio.Queue], loop: asyncio.AbstractEventLoop, agent_role: str):
    """
    Creates a list of executable Python functions (wrappers) for Swarm from ToolBench summaries.
    Hooks execution into the adapter and optionally notifies a queue for UI streaming.
    """
    funcs = []
    
    def make_wrapper(tool_def):
        tool_name = tool_def["name"]
        
        def wrapper(**kwargs):
            # 1. Notify UI of Start (if queue provided)
            if queue and loop:
                # Swarm doesn't yield "Call: ..." explicitly before execution in stream
                # So we push a thought/event here
                loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "thought",
                            "agent": agent_role,
                            "content": f"\n\n[Action: {tool_name}]\n"
                        }
                )

            action_obj = {"tool": tool_name, "arguments": kwargs}
            
            # 2. Execute via Adapter
            # The adapter handles cache lookup or fake generation
            obs_result = adapter.step(
                action_obj,
                tool_summaries,
                state=task_state
            )
            
            # 3. Notify UI of Step Completion (if queue provided)
            # This allows the UI to render the 'Action Card' efficiently
            if queue and loop:
                step_data = {
                    "agent": agent_role,
                    "thought": f"Calling {tool_name}...", 
                    "action": f"Call: {tool_name}",
                    "observation": "", # observation comes in next 'tool' message
                    "done": False,
                    "final_answer": "",
                    "distribution": [], # Swarm Native: No probs
                    "stepIndex": -1, # Will be fixed by caller or ignored
                    "metrics": {"latency": 0.0, "tokens": 0.0}
                }
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "type": "result", 
                        "data": step_data
                    }
                )

            return obs_result["observation"]

        wrapper.__name__ = tool_name
        wrapper.__doc__ = tool_def.get("description", "")
        return wrapper

    for t in tool_summaries:
        funcs.append(make_wrapper(t))
        
    return funcs

# --- Generic Extractors (Shared) ---
import json
import re

try:
    from agentmark.sdk.prompt_adapter import extract_json_payload
except ImportError:
    # Fallback if path not set yet or running standalone test
    def extract_json_payload(text): return {}

def _extract_watermark(completion: Any) -> Optional[Dict[str, Any]]:
    try:
        extra = getattr(completion, "model_extra", None)
        if extra and isinstance(extra, dict) and extra.get("watermark"):
            return extra.get("watermark")
        extra = getattr(completion, "__pydantic_extra__", None)
        if extra and isinstance(extra, dict) and extra.get("watermark"):
            return extra.get("watermark")
        # Try model dump
        if hasattr(completion, "model_dump"):
            payload = completion.model_dump()
            return payload.get("watermark")
        return None
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
