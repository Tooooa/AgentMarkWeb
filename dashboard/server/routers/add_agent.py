
import json
import time
import os
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from agentmark.sdk.prompt_adapter import get_prompt_instruction

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
            "fallback_actions": candidates
        }

        use_swarm = (os.getenv("AGENTMARK_USE_SWARM") or "1").strip().lower() not in {"0", "false", "no"}
        model_name = session.model
        
        steps = []
        w_latency = 0.0
        w_start = time.time()
        
        prompt_trace = None
        
        if use_swarm:
            try:
                from swarm import Swarm, Agent
                
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
                
                # Initialize Adapter
                from agentmark.environments.toolbench.adapter import ToolBenchAdapter
                adapter = ToolBenchAdapter(TOOL_DATA_ROOT, client=base_client) # Adapter can use base client
                
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
                response = swarm_client.run(
                    agent=ephemeral_agent,
                    messages=[{"role": "user", "content": req.message.strip()}],
                    context_variables={},
                    model_override=model_name,
                    stream=False,
                    debug=True,
                    execute_tools=True
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
                print(f"[WARN] AddAgent Swarm bridge failed: {exc}")
                import traceback
                traceback.print_exc()
                raise exc

        else:
            # Fallback to single turn legacy
            client = _build_proxy_client(req.apiKey or session.api_key)
            w_completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=ADD_AGENT_TOOLS, 
                extra_body={"agentmark": agentmark_body},
                temperature=0.0,
            )
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

    raw_text = watermark.get("raw_llm_output") or ""
    thought = _extract_thought_from_raw_output(raw_text)
    if not thought and isinstance(completion_content, str):
         thought = _extract_thought_from_raw_output(completion_content)
         
    if not thought:
        if completion_content and not completion_tool_calls:
             pass 
        else:
             thought = "Thinking..."

    bits_embedded = frontend.get("watermark_meta", {}).get("bits_embedded") or 0
    matrix_rows = [[1] for _ in range(int(bits_embedded))]
    
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
