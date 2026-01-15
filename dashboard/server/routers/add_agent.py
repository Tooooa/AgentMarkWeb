
import json
import time
import os
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from agentmark.sdk.prompt_adapter import get_prompt_instruction, extract_json_payload
from openai import APIConnectionError, APIStatusError, APITimeoutError

# Import shared utilities
from dashboard.server.shared import (
    _resolve_api_key, _get_base_llm_base, _get_proxy_base, _build_proxy_client,
    _create_dynamic_swarm_tools, TOOL_DATA_ROOT,
    _extract_watermark, _extract_tokens_used, _extract_thought_from_raw_output
)

router = APIRouter(prefix="/api/add_agent", tags=["add_agent"])

# --- Models ---

class AddAgentInitRequest(BaseModel):
    modelUrl: str
    apiKey: Optional[str] = None

class AddAgentTurnRequest(BaseModel):
    sessionId: str
    message: str
    apiKey: Optional[str] = None

class SaveAddAgentSessionRequest(BaseModel):
    sessionId: str
    name: str

# --- Constants & Tool Definitions ---

ADD_AGENT_SYSTEM_PROMPT = "You are a helpful agent."

def get_weather(location: str, time: str = "now") -> str:
    """Get the current weather in a given location. Location MUST be a city."""
    return json.dumps({"location": location, "temperature": "65", "time": time})

def get_weather_forecast(location: str, days: str = "3") -> str:
    """Get a short weather forecast for a given location and number of days."""
    try:
        days_val = int(days)
    except Exception:
        days_val = 3
    return json.dumps(
        {"location": location, "days": days_val, "forecast": ["sunny", "cloudy", "rain"]}
    )

def get_air_quality(location: str) -> str:
    """Get a simple air quality report for a given location."""
    return json.dumps({"location": location, "aqi": 42, "status": "good"})

def send_email(recipient: str, subject: str, body: str) -> str:
    """Send a short email."""
    print("Sending email...")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return "Sent!"

def send_sms(phone_number: str, message: str) -> str:
    """Send a short SMS message to a phone number."""
    print("Sending sms...")
    print(f"To: {phone_number}")
    print(f"Message: {message}")
    return "Sent!"

def get_top_rated_movies(limit: int = 10, min_imdb: float = 8.0) -> str:
    """Return a list of top-rated movies with IMDb scores."""
    return json.dumps(
        {
            "limit": limit,
            "min_imdb": min_imdb,
            "results": [
                {"title": "The Shawshank Redemption", "imdb": 9.3},
                {"title": "The Godfather", "imdb": 9.2},
                {"title": "The Dark Knight", "imdb": 9.0},
            ],
        }
    )

def search_movies_by_genre(genre: str, limit: int = 10) -> str:
    """Search movies by genre."""
    return json.dumps(
        {
            "genre": genre,
            "limit": limit,
            "results": ["Inception", "Interstellar", "The Matrix"],
        }
    )

def get_movie_summary(title: str) -> str:
    """Fetch a short summary for a movie title."""
    return json.dumps(
        {
            "title": title,
            "summary": "A brief synopsis for the requested movie.",
        }
    )

def search_web(query: str) -> str:
    """Search the web for general queries."""
    return json.dumps({"query": query, "results": []})

ADD_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location. Location MUST be a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}, "time": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get a short weather forecast for a given location and number of days.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}, "days": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": "Get a simple air quality report for a given location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["recipient", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send a short SMS message to a phone number.",
            "parameters": {
                "type": "object",
                "properties": {"phone_number": {"type": "string"}, "message": {"type": "string"}},
                "required": ["phone_number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_rated_movies",
            "description": "Return a list of top-rated movies with IMDb scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "min_imdb": {"type": "number"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_movies_by_genre",
            "description": "Search movies by genre.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["genre"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_summary",
            "description": "Fetch a short summary for a movie title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for general queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]

# --- Helpers ---

def _resolve_base_model(target_model_env: str) -> str:
    # Use existing map logic or simplified
    _BASE_MODEL_MAP = {
        "gpt-4o": "deepseek-chat",
        "gpt-4o-mini": "deepseek-chat",
        "gpt-4-turbo": "deepseek-chat",
        "gpt-4": "deepseek-chat",
        "gpt-3.5-turbo": "deepseek-chat",
    }
    return _BASE_MODEL_MAP.get(target_model_env, "deepseek-chat")

def _get_add_agent_candidates() -> List[str]:
    candidates: List[str] = []
    for tool in ADD_AGENT_TOOLS:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {})
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if name and name not in candidates:
            candidates.append(name)
    return candidates

def _maybe_parse_agentmark_payload(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    try:
        payload = extract_json_payload(stripped)
    except Exception:
        return None

    if not isinstance(payload, dict) or not payload:
        return None

    if any(key in payload for key in ("action_weights", "action_probs", "scores", "action_args", "thought")):
        return payload
    return None

def _payload_to_distribution(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    weights = payload.get("action_weights") or payload.get("action_probs") or payload.get("scores")
    if not isinstance(weights, dict):
        return []
    distribution: List[Dict[str, Any]] = []
    for name, raw_val in weights.items():
        try:
            prob = float(raw_val)
        except Exception:
            continue
        distribution.append({"name": str(name), "prob": prob, "isSelected": False})
    return distribution

def _choose_action_from_payload(payload: Dict[str, Any], distribution: List[Dict[str, Any]]) -> str:
    direct = payload.get("action") or payload.get("tool")
    if direct:
        return str(direct)
    if not distribution:
        return ""
    return max(distribution, key=lambda item: float(item.get("prob") or 0.0)).get("name") or ""

def _extract_action_args_from_payload(payload: Dict[str, Any], action: str) -> Dict[str, Any]:
    raw_args = payload.get("action_args")
    if not isinstance(raw_args, dict):
        return {}
    if action and action in raw_args and isinstance(raw_args.get(action), dict):
        return raw_args.get(action) or {}
    return raw_args

def _build_add_agent_scoring_messages(user_message: str) -> List[Dict[str, str]]:
    instr = get_prompt_instruction()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": ADD_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]

    candidates = _get_add_agent_candidates()
    if candidates:
        user_lines = "候选动作：\\n" + "\\n".join(f"- {c}" for c in candidates)
        tool_lines = ""
        tool_specs = []
        for tool in ADD_AGENT_TOOLS:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", {}) if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name") or tool.get("name")
            params = fn.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            if name:
                if props:
                    keys = ", ".join(props.keys())
                    tool_specs.append(f"- {name}({keys})")
                else:
                    tool_specs.append(f"- {name}(...)")
        if tool_specs:
            tool_lines = "\\n可用工具参数：\\n" + "\\n".join(tool_specs)
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] = (msg["content"] or "") + "\\n" + user_lines + tool_lines
                break
        else:
            messages.append({"role": "user", "content": user_lines + tool_lines})

    injected = [{"role": "system", "content": instr}]
    injected.extend(messages)
    injected[0]["content"] += "\\n[AgentMark mode=tools]"
    return injected

def _coerce_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    return getattr(obj, "__dict__", str(obj))

def _extract_stream_delta(ev: Any) -> Dict[str, Any]:
    payload = _coerce_to_dict(ev)
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0] if isinstance(choices[0], dict) else {}
    delta = first.get("delta")
    return delta if isinstance(delta, dict) else {}

def _extract_partial_thought(buf: str) -> Optional[str]:
    if not buf:
        return None
    lowered = buf.lower()
    idx = lowered.rfind('"thought"')
    quote = '"'
    if idx == -1:
        idx = lowered.rfind("'thought'")
        quote = "'"
    if idx == -1:
        return None

    after = buf[idx + len(quote + "thought" + quote):]
    colon = after.find(":")
    if colon == -1:
        return None
    rest = after[colon + 1:].lstrip()
    if not rest.startswith(("\"", "'")):
        return None
    q = rest[0]
    rest = rest[1:]

    out = []
    escaped = False
    for ch in rest:
        if escaped:
            if ch == "n":
                out.append("\n")
            elif ch == "t":
                out.append("\t")
            else:
                out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == q:
            break
        out.append(ch)

    candidate = "".join(out)
    return candidate if candidate.strip() else None

def _build_swarm_system_prompt() -> str:
    """Builds the system instructions including the JSON requirement."""
    instr = get_prompt_instruction()
    candidates = _get_add_agent_candidates()
    
    prompt = ADD_AGENT_SYSTEM_PROMPT + "\\n\\n" + instr
    
    if candidates:
        prompt += "\\n\\n候选动作：\\n" + "\\n".join(f"- {c}" for c in candidates)
        
        tool_specs = []
        for tool in ADD_AGENT_TOOLS:
            fn = tool.get("function", {}) if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name") or tool.get("name")
            params = fn.get("parameters", {}) or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            if name:
                if props:
                    keys = ", ".join(props.keys())
                    tool_specs.append(f"- {name}({keys})")
                else:
                    tool_specs.append(f"- {name}(...)")
        
        if tool_specs:
            prompt += "\\n\\n可用工具参数：\\n" + "\\n".join(tool_specs)
            
    prompt += "\\n\\n[AgentMark mode=tools]"
    return prompt

# --- Session & State ---

class AddAgentSession:
    def __init__(self, session_id: str, api_key: str, repo_url: str):
        self.session_id = session_id
        self.api_key = api_key
        self.repo_url = repo_url
        self.step_count = 0
        self.task_query = ""
        self.last_user_message = ""
        self.watermarked_trajectory: List[Dict[str, str]] = []
        self.baseline_trajectory: List[Dict[str, str]] = []
        self.model = _resolve_base_model(os.getenv("AGENTMARK_TARGET_MODEL", "gpt-4o"))
        self.async_client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "anything",
            base_url=_get_base_llm_base(),
        )

add_agent_sessions: Dict[str, AddAgentSession] = {}

# --- Client Wrapper ---

class AgentMarkSwarmClientWrapper:
    """
    Wraps the OpenAI client to:
    1. Inject 'agentmark' extra_body (candidates, phase) into every completion call.
    2. Capture the exact raw completion object to extract watermark/prompt data later.
    """
    def __init__(self, client: OpenAI, agentmark_body: Dict[str, Any]):
        self.client = client
        self.agentmark_body = agentmark_body
        self.last_completion = None
        
        # We need to mock client.chat.completions.create
        # So we create a nest of objects that Swarm will traverse.
        class Completions:
            def __init__(self, wrapper):
                self.wrapper = wrapper
            def create(self, **kwargs):
                return self.wrapper._create(**kwargs)
        
        class Chat:
            def __init__(self, wrapper):
                self.completions = Completions(wrapper)
                
        self.chat = Chat(self)

    def _create(self, **kwargs):
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
        # Merge our agentmark data
        if "agentmark" not in kwargs["extra_body"]:
             kwargs["extra_body"]["agentmark"] = {}
        kwargs["extra_body"]["agentmark"].update(self.agentmark_body)
        
        # Call the real client
        print(f"[Wrapper] Calling upstream with agentmark payload: {self.agentmark_body.keys()}")
        try:
            completion = self.client.chat.completions.create(**kwargs)
            self.last_completion = completion
            return completion
        except Exception as e:
            print(f"[Wrapper] Upstream call failed: {e}")
            raise e


# --- Endpoints ---

@router.post("/init")
async def init_session(req: AddAgentInitRequest):
    session_id = str(uuid.uuid4())
    api_key = _resolve_api_key(req.apiKey)
    
    session = AddAgentSession(session_id, api_key, req.modelUrl)
    add_agent_sessions[session_id] = session
    
    return {
        "sessionId": session_id,
        "status": "initialized",
        "model": session.model
    }

@router.post("/turn")
async def add_agent_turn(req: AddAgentTurnRequest):
    try:
        if req.sessionId not in add_agent_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = add_agent_sessions[req.sessionId]
        session.last_user_message = req.message
        
        # 1. Prepare Data for Watermarking (Proxy)
        # We need to give the proxy enough info to score probabilities.
        # Although Swarm handles the 'execution' messages, we pass the *user's initial intent* 
        # and candidates to the proxy via extra_body.
        messages = _build_add_agent_scoring_messages(req.message)
        candidates = _get_add_agent_candidates()
        
        # This body is injected by the wrapper
        agentmark_body = {
            "messages": messages, 
            "phase": "score", 
            "model": session.model,
            "candidates": candidates # Corrected key
        }

        use_swarm = (os.getenv("AGENTMARK_USE_SWARM") or "1").strip().lower() not in {"0", "false", "no"}
        model_name = session.model
        
        steps = []
        w_latency = 0.0
        w_start = time.time()
        
        prompt_trace = None
        
        if use_swarm:
            try:
                try:
                    from swarm import Swarm, Agent
                except Exception as swarm_exc:
                    print(f"[WARN] Swarm import failed, falling back to non-swarm mode: {swarm_exc}")
                    use_swarm = False
                    Swarm = None  # type: ignore[assignment]
                    Agent = None  # type: ignore[assignment]
                    raise
                
                # Create the wrapped client
                base_client = _build_proxy_client(req.apiKey or session.api_key)
                wrapped_client = AgentMarkSwarmClientWrapper(base_client, agentmark_body)
                
                swarm_client = Swarm(client=wrapped_client)
                
                # --- Authentic ToolBench Setup for Plugin Mode ---
                plugin_tool_defs = [
                    {"name": "search_web", "description": "Search the web for general queries.", "arguments": {"query": "string"}},
                    {"name": "get_weather", "description": "Get weather for a location.", "arguments": {"location": "string"}},
                    {"name": "get_weather_forecast", "description": "Get weather forecast.", "arguments": {"location": "string"}},
                    {"name": "get_air_quality", "description": "Get air quality.", "arguments": {"location": "string"}},
                    {"name": "send_email", "description": "Send an email.", "arguments": {"recipient": "string", "subject": "string", "body": "string"}},
                    {"name": "send_sms", "description": "Send an SMS.", "arguments": {"phone_number": "string", "message": "string"}},
                    {"name": "get_top_rated_movies", "description": "Get top rated movies.", "arguments": {"limit": "integer"}},
                    {"name": "search_movies_by_genre", "description": "Search movies by genre.", "arguments": {"genre": "string"}},
                    {"name": "get_movie_summary", "description": "Get movie summary.", "arguments": {"title": "string"}},
                ]
                
                # Initialize Adapter - Use DIRECT DeepSeek client, NOT proxy
                # Fake response generation should NOT go through watermark proxy
                from openai import OpenAI as DirectOpenAI
                direct_client = DirectOpenAI(
                    api_key=req.apiKey or session.api_key,
                    base_url="https://api.deepseek.com"
                )
                from agentmark.environments.toolbench.adapter import ToolBenchAdapter
                adapter = ToolBenchAdapter(TOOL_DATA_ROOT, client=direct_client) # Direct to DeepSeek
                
                loop = asyncio.get_running_loop()
                funcs = _create_dynamic_swarm_tools(
                    plugin_tool_defs,
                    adapter,
                    {"query": req.message},
                    None,
                    loop,
                    "AddAgent"
                )
                
                # Build SYSTEM PROMPT that forces JSON
                system_instructions = _build_swarm_system_prompt()

                ephemeral_agent = Agent(
                    name="AddAgent", 
                    model=model_name, 
                    instructions=system_instructions,
                    functions=funcs
                )
                
                # Run Swarm
                # Swarm's run returns the updated conversation history (including input messages)
                # max_turns limits tool call iterations to prevent infinite loops
                response = swarm_client.run(
                    agent=ephemeral_agent,
                    messages=[{"role": "user", "content": req.message.strip()}],
                    context_variables={},
                    model_override=model_name,
                    stream=False,
                    debug=True,
                    execute_tools=True,
                    max_turns=5  # Limit to 5 tool call iterations
                )
                
                # --- Recover Prompt Trace & Watermark from the Wrapper ---
                if wrapped_client.last_completion:
                    try:
                        # Extract prompt trace
                        model_extra = getattr(wrapped_client.last_completion, "model_extra", {}) or {}
                        # Check __pydantic_extra__ if needed
                        if not model_extra:
                             model_extra = getattr(wrapped_client.last_completion, "__pydantic_extra__", {}) or {}

                        if "prompt_trace" in model_extra:
                            prompt_trace = model_extra["prompt_trace"]
                        
                        # We also want to ensure we have the watermark for the LAST step
                        # Swarm might have done multiple steps. 
                        # Ideally we'd capture ALL completions.
                        # For now, let's assume the user cares about the final or most recent one for display.
                    except Exception as ex:
                        print(f"[WARN] Failed to extract trace from wrapper: {ex}")

                # --- Process History to Build Steps ---
                # We want to identify the NEW steps generated in this turn.
                new_messages = response.messages[1:] # Skip the user message we just sent
                
                # Grouping Logic: Assistant (Call) -> Tool (Result)
                i = 0
                while i < len(new_messages):
                    msg = new_messages[i]
                    role = msg.get("role")
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")
                    
                    if role == "assistant":
                        session.step_count += 1
                        
                        observation = None
                        if tool_calls:
                            next_msg_idx = i + 1
                            if next_msg_idx < len(new_messages):
                                 next_msg = new_messages[next_msg_idx]
                                 if next_msg.get("role") == "tool":
                                     observation = next_msg.get("content")
                                     i += 1 
                        
                        # Extract watermark from the MSG dict if Swarm preserved it?
                        # Swarm converts message objects to dicts, so custom Pydantic fields might be lost
                        # UNLESS we manually grab it from the wrapper? 
                        # But wrapper.last_completion only has the LAST one.
                        # If there were multiple steps, we only have the last one's metadata.
                        # This is a limitation of this hotfix.
                        # However, Swarm 'messages' dicts usually just copy the standard fields.
                        
                        # Attempt to use the wrapper's data for at least one step
                        watermark_data = {}
                        if wrapped_client.last_completion:
                             w = _extract_watermark(wrapped_client.last_completion)
                             if w:
                                 watermark_data = w
                        
                        step_data = _build_add_agent_step(
                            watermark=watermark_data,
                            step_index=session.step_count,
                            completion_content=content,
                            completion_tool_calls=tool_calls,
                            observation=observation,
                            latency=0.0, 
                            tokens=_extract_tokens_used(msg)
                        )
                        steps.append(step_data)
                    
                    i += 1
                    
                w_latency = time.time() - w_start

            except Exception as exc:
                if not use_swarm:
                    # Continue with non-swarm fallback below
                    pass
                else:
                    print(f"[WARN] AddAgent Swarm bridge failed: {exc}")
                    import traceback
                    traceback.print_exc()
                    raise exc

        if not use_swarm:
            # Fallback to single turn legacy
            resolved_key = req.apiKey or session.api_key
            proxy_client = _build_proxy_client(resolved_key)
            direct_client = OpenAI(api_key=resolved_key, base_url=_get_base_llm_base())

            def is_retryable(err: Exception) -> bool:
                if isinstance(err, (APIConnectionError, APITimeoutError)):
                    return True
                if isinstance(err, APIStatusError):
                    return int(getattr(err, "status_code", 0) or 0) in {429, 500, 502, 503, 504}
                return False

            w_completion = None
            last_exc: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    w_completion = proxy_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=ADD_AGENT_TOOLS,
                        extra_body={"agentmark": agentmark_body},
                        temperature=0.0,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 3 and is_retryable(exc):
                        await asyncio.sleep(0.6 * (2 ** (attempt - 1)))
                        continue
                    # Fall back to direct client without extra_body (proxy may be down / incompatible).
                    w_completion = direct_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=ADD_AGENT_TOOLS,
                        temperature=0.0,
                    )
                    last_exc = None
                    break

            if w_completion is None:
                raise last_exc or RuntimeError("Completion failed")
            w_latency = time.time() - w_start
            msg = w_completion.choices[0].message
            
            session.step_count += 1
            
            # Extract Trace
            model_extra = getattr(w_completion, "model_extra", {}) or {}
            prompt_trace = model_extra.get("prompt_trace")

            step_data = _build_add_agent_step(
                watermark=_extract_watermark(w_completion) or {},
                step_index=session.step_count,
                completion_content=msg.content,
                completion_tool_calls=msg.tool_calls,
                observation=None,
                latency=w_latency,
                tokens=_extract_tokens_used(w_completion)
            )
            steps.append(step_data)

        return {
            "steps": steps, 
            "step": steps[-1] if steps else None,
            "promptTrace": prompt_trace, 
            "baselinePromptTrace": None
        }
    except Exception as e:
        import traceback
        with open("/root/autodl-tmp/AgentMark2/AgentMark/dashboard/error.log", "a") as f:
            f.write(f"\\n[CRITICAL ERROR] add_agent_turn failed: {e}\\n")
            f.write(traceback.format_exc())
            f.write("\\n")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/turn_stream")
async def add_agent_turn_stream(req: AddAgentTurnRequest):
    if req.sessionId not in add_agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = add_agent_sessions[req.sessionId]
    session.last_user_message = req.message

    messages = _build_add_agent_scoring_messages(req.message)
    candidates = _get_add_agent_candidates()
    agentmark_body = {
        "messages": messages,
        "phase": "score",
        "model": session.model,
        "candidates": candidates,
    }

    use_swarm = (os.getenv("AGENTMARK_USE_SWARM") or "1").strip().lower() not in {"0", "false", "no"}
    model_name = session.model

    async def event_gen():
        yield json.dumps({"type": "status", "data": {"state": "start"}}) + "\n"

        try:
            swarm_enabled = use_swarm
            swarm_import_error = None
            if swarm_enabled:
                try:
                    from swarm import Swarm, Agent
                except Exception as swarm_exc:
                    swarm_enabled = False
                    swarm_import_error = str(swarm_exc)
                    yield json.dumps({"type": "status", "data": {"state": "swarm_unavailable", "reason": swarm_import_error}}) + "\n"

            if swarm_enabled:
                base_client = _build_proxy_client(req.apiKey or session.api_key)
                wrapped_client = AgentMarkSwarmClientWrapper(base_client, agentmark_body)
                swarm_client = Swarm(client=wrapped_client)

                plugin_tool_defs = [
                    {"name": "search_web", "description": "Search the web for general queries.", "arguments": {"query": "string"}},
                    {"name": "get_weather", "description": "Get weather for a location.", "arguments": {"location": "string"}},
                    {"name": "get_weather_forecast", "description": "Get weather forecast.", "arguments": {"location": "string"}},
                    {"name": "get_air_quality", "description": "Get air quality.", "arguments": {"location": "string"}},
                    {"name": "send_email", "description": "Send an email.", "arguments": {"recipient": "string", "subject": "string", "body": "string"}},
                    {"name": "send_sms", "description": "Send an SMS.", "arguments": {"phone_number": "string", "message": "string"}},
                    {"name": "get_top_rated_movies", "description": "Get top rated movies.", "arguments": {"limit": "integer"}},
                    {"name": "search_movies_by_genre", "description": "Search movies by genre.", "arguments": {"genre": "string"}},
                    {"name": "get_movie_summary", "description": "Get movie summary.", "arguments": {"title": "string"}},
                ]

                from openai import OpenAI as DirectOpenAI
                direct_client = DirectOpenAI(
                    api_key=req.apiKey or session.api_key,
                    base_url="https://api.deepseek.com",
                )
                from agentmark.environments.toolbench.adapter import ToolBenchAdapter
                adapter = ToolBenchAdapter(TOOL_DATA_ROOT, client=direct_client)

                loop = asyncio.get_running_loop()
                funcs = _create_dynamic_swarm_tools(
                    plugin_tool_defs,
                    adapter,
                    {"query": req.message},
                    None,
                    loop,
                    "AddAgent",
                )

                system_instructions = _build_swarm_system_prompt()
                ephemeral_agent = Agent(
                    name="AddAgent",
                    model=model_name,
                    instructions=system_instructions,
                    functions=funcs,
                )

                partial_buf = ""
                last_sent_thought = None
                last_tool_name = None
                last_tool_args = None
                final_response = None

                stream = swarm_client.run(
                    agent=ephemeral_agent,
                    messages=[{"role": "user", "content": req.message.strip()}],
                    context_variables={},
                    model_override=model_name,
                    stream=True,
                    debug=True,
                    execute_tools=True,
                    max_turns=5,
                )

                for ev in stream:
                    payload = _coerce_to_dict(ev)
                    if isinstance(payload, dict) and "response" in payload:
                        final_response = payload.get("response")
                        continue

                    if isinstance(payload, dict) and isinstance(payload.get("delim"), str):
                        yield json.dumps({"type": "delim", "data": payload.get("delim")}) + "\n"
                        continue

                    delta = _extract_stream_delta(payload)
                    if delta:
                        content_delta = delta.get("content")
                        if isinstance(content_delta, str) and content_delta:
                            partial_buf += content_delta
                            thought = _extract_partial_thought(partial_buf) or _extract_thought_from_raw_output(partial_buf)
                            if thought and thought != last_sent_thought:
                                last_sent_thought = thought
                                yield json.dumps({"type": "thought_delta", "data": {"text": thought}}) + "\n"

                        tool_calls = delta.get("tool_calls")
                        if isinstance(tool_calls, list) and tool_calls:
                            first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
                            fn = first.get("function") if isinstance(first, dict) else None
                            fn = fn if isinstance(fn, dict) else {}
                            tool_name = fn.get("name")
                            tool_args = fn.get("arguments")
                            if tool_name and (tool_name != last_tool_name or tool_args != last_tool_args):
                                last_tool_name = tool_name
                                last_tool_args = tool_args
                                yield json.dumps({"type": "tool_call", "data": {"name": tool_name, "arguments": tool_args}}) + "\n"

                response_obj = final_response
                response_dict = _coerce_to_dict(response_obj)
                response_messages = None
                if hasattr(response_obj, "messages"):
                    response_messages = getattr(response_obj, "messages", None)
                if response_messages is None and isinstance(response_dict, dict):
                    response_messages = response_dict.get("messages")

                steps: List[Dict[str, Any]] = []
                if isinstance(response_messages, list):
                    new_messages = response_messages[1:]
                    i = 0
                    while i < len(new_messages):
                        msg = new_messages[i] if isinstance(new_messages[i], dict) else {}
                        role = msg.get("role")
                        content = msg.get("content")
                        tool_calls = msg.get("tool_calls")

                        if role == "assistant":
                            session.step_count += 1

                            observation = None
                            if tool_calls:
                                next_msg_idx = i + 1
                                if next_msg_idx < len(new_messages):
                                    next_msg = new_messages[next_msg_idx] if isinstance(new_messages[next_msg_idx], dict) else {}
                                    if next_msg.get("role") == "tool":
                                        observation = next_msg.get("content")
                                        i += 1

                            watermark_data = {}
                            step_data = _build_add_agent_step(
                                watermark=watermark_data,
                                step_index=session.step_count,
                                completion_content=content,
                                completion_tool_calls=tool_calls,
                                observation=observation,
                                latency=0.0,
                                tokens=_extract_tokens_used(msg),
                            )
                            steps.append(step_data)
                        i += 1

                yield json.dumps({
                    "type": "result",
                    "data": {
                        "steps": steps,
                        "step": steps[-1] if steps else None,
                        "promptTrace": None,
                        "baselinePromptTrace": None,
                    },
                }) + "\n"
            else:
                # Non-swarm: best-effort streaming of the LLM content. We still emit thought/tool deltas,
                # then emit a final Step built from the accumulated content.
                resolved_key = req.apiKey or session.api_key
                proxy_client = _build_proxy_client(resolved_key)
                direct_client = OpenAI(api_key=resolved_key, base_url=_get_base_llm_base())
                partial_buf = ""
                last_sent_thought = None
                tool_call_buf: List[Dict[str, Any]] = []

                def is_retryable(err: Exception) -> bool:
                    if isinstance(err, (APIConnectionError, APITimeoutError)):
                        return True
                    if isinstance(err, APIStatusError):
                        return int(getattr(err, "status_code", 0) or 0) in {429, 500, 502, 503, 504}
                    return False

                def create_stream(client: OpenAI, *, use_agentmark: bool):
                    kwargs: Dict[str, Any] = {
                        "model": model_name,
                        "messages": messages,
                        "tools": ADD_AGENT_TOOLS,
                        "temperature": 0.0,
                        "stream": True,
                    }
                    if use_agentmark:
                        kwargs["extra_body"] = {"agentmark": agentmark_body}
                    return client.chat.completions.create(**kwargs)

                # Prefer proxy (for agentmark extra_body). If proxy isn't reachable, fall back to direct.
                stream = None
                use_agentmark = True
                try:
                    stream = create_stream(proxy_client, use_agentmark=True)
                except Exception as first_exc:
                    use_agentmark = False
                    yield json.dumps({"type": "status", "data": {"state": "proxy_unavailable", "reason": str(first_exc)}}) + "\n"
                    stream = create_stream(direct_client, use_agentmark=False)

                max_attempts = 3
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        for ev in stream:
                            payload = _coerce_to_dict(ev)
                            delta = _extract_stream_delta(payload)
                            if not delta:
                                continue
                            content_delta = delta.get("content")
                            if isinstance(content_delta, str) and content_delta:
                                partial_buf += content_delta
                                thought = _extract_partial_thought(partial_buf) or _extract_thought_from_raw_output(partial_buf)
                                if thought and thought != last_sent_thought:
                                    last_sent_thought = thought
                                    yield json.dumps({"type": "thought_delta", "data": {"text": thought}}) + "\n"

                            tool_calls = delta.get("tool_calls")
                            if isinstance(tool_calls, list) and tool_calls:
                                tool_call_buf = tool_calls
                                first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
                                fn = first.get("function") if isinstance(first, dict) else None
                                fn = fn if isinstance(fn, dict) else {}
                                tool_name = fn.get("name")
                                tool_args = fn.get("arguments")
                                if tool_name:
                                    yield json.dumps({"type": "tool_call", "data": {"name": tool_name, "arguments": tool_args}}) + "\n"
                        break
                    except Exception as stream_exc:
                        if attempt >= max_attempts or not is_retryable(stream_exc):
                            raise
                        wait_s = 0.6 * (2 ** (attempt - 1))
                        yield json.dumps({"type": "status", "data": {"state": "retrying", "attempt": attempt, "reason": str(stream_exc)}}) + "\n"
                        await asyncio.sleep(wait_s)
                        partial_buf = ""
                        last_sent_thought = None
                        tool_call_buf = []
                        # If the proxy path is flaky, fall back to direct on retry.
                        if use_agentmark and isinstance(stream_exc, (APIConnectionError, APITimeoutError, APIStatusError)):
                            use_agentmark = False
                            yield json.dumps({"type": "status", "data": {"state": "proxy_fallback", "reason": str(stream_exc)}}) + "\n"
                        stream = create_stream(proxy_client, use_agentmark=True) if use_agentmark else create_stream(direct_client, use_agentmark=False)

                session.step_count += 1
                step_data = _build_add_agent_step(
                    watermark={},
                    step_index=session.step_count,
                    completion_content=partial_buf,
                    completion_tool_calls=tool_call_buf or None,
                    observation=None,
                    latency=0.0,
                    tokens=0.0,
                )
                yield json.dumps({
                    "type": "result",
                    "data": {
                        "steps": [step_data],
                        "step": step_data,
                        "promptTrace": None,
                        "baselinePromptTrace": None,
                    },
                }) + "\n"

        except Exception as e:
            msg = str(e)
            status_code = None
            if isinstance(e, APIStatusError):
                status_code = int(getattr(e, "status_code", 0) or 0) or None
                msg = f"Upstream HTTP {status_code}: {msg}" if status_code else msg
                if status_code == 503:
                    msg += " (Service Unavailable: upstream overloaded/down; please retry later.)"
            if isinstance(e, (APIConnectionError, APITimeoutError)):
                msg = f"Upstream connection failed: {msg}"
            yield json.dumps({"type": "error", "message": msg, "status_code": status_code}) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")

@router.post("/save")
async def save_add_agent_session(req: SaveAddAgentSessionRequest):
    if req.sessionId not in add_agent_sessions:
         raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "saved", "id": req.sessionId}

def _build_add_agent_step(
    watermark: Dict[str, Any],
    step_index: int,
    completion_content: Optional[str],
    completion_tool_calls: Optional[List[Dict[str, Any]]],
    *,
    observation: Optional[str],
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

    agentmark_payload = _maybe_parse_agentmark_payload(completion_content)
    if not agentmark_payload:
        raw_text_candidate = watermark.get("raw_llm_output") or ""
        agentmark_payload = _maybe_parse_agentmark_payload(raw_text_candidate)
    is_agentmark_scoring = bool(agentmark_payload)
    
    if completion_tool_calls:
        tc = completion_tool_calls[0]
        if hasattr(tc, 'function'):
            fn_name = tc.function.name
            fn_args = tc.function.arguments
        else:
            fn = tc.get('function', {})
            fn_name = fn.get('name', 'unknown_tool')
            fn_args = fn.get('arguments', '{}')
            
        action = fn_name
        try:
            action_args = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
        except:
            action_args = {"raw_args": fn_args}
    elif is_agentmark_scoring:
        parsed_distribution = _payload_to_distribution(agentmark_payload or {})
        parsed_action = _choose_action_from_payload(agentmark_payload or {}, parsed_distribution)
        parsed_args = _extract_action_args_from_payload(agentmark_payload or {}, parsed_action)

        if not distribution and parsed_distribution:
            distribution = parsed_distribution

        if parsed_action:
            for item in distribution:
                if item.get("name") == parsed_action:
                    item["isSelected"] = True
            action = action or parsed_action

        if parsed_args:
            action_args = parsed_args

    raw_text = watermark.get("raw_llm_output") or ""
    thought = _extract_thought_from_raw_output(raw_text)
    if not thought and isinstance(completion_content, str):
         thought = _extract_thought_from_raw_output(completion_content)
         
    if not thought:
        if completion_tool_calls and isinstance(completion_content, str) and completion_content.strip():
            candidate = completion_content.strip()
            if not candidate.startswith("{"):
                thought = candidate
        if not thought:
            if completion_content and not completion_tool_calls:
                if is_agentmark_scoring:
                    thought = "Thinking..."
            else:
                thought = "Thinking..."

    bits_embedded = frontend.get("watermark_meta", {}).get("bits_embedded") or 0
    matrix_rows = [[1] for _ in range(int(bits_embedded))]
    
    final_answer = ""
    if is_agentmark_scoring and str(action).strip() == "Finish" and not completion_tool_calls:
        if isinstance(action_args, dict):
            final_answer = (action_args.get("final_answer") or action_args.get("answer") or "").strip()
        elif isinstance(action_args, str):
            final_answer = action_args.strip()

    if not final_answer and not is_agentmark_scoring and not completion_tool_calls:
        final_answer = (completion_content or "").strip()
        if not final_answer and raw_text and not completion_tool_calls:
            final_answer = raw_text.strip()
    
    has_tool_calls = bool(completion_tool_calls)
    step_type = "tool" if (action or has_tool_calls) else "other"
    if final_answer and not has_tool_calls:
        step_type = "finish"

    tool_details_display = ""
    if observation:
        tool_details_display = observation
    elif action_args and not observation:
        tool_details_display = f"Input arguments: {json.dumps(action_args, ensure_ascii=False)}"
    
    return {
        "stepIndex": step_index,
        "thought": thought,
        "action": f"Call: {action}" if action else "",
        "toolDetails": tool_details_display,
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
