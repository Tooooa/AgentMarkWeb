
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
# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SWARM_ROOT = PROJECT_ROOT / "swarm"
TOOL_DATA_ROOT = PROJECT_ROOT / "experiments/toolbench/data/data/toolenv/tools"


def _load_root_dotenv() -> None:
    """尽力加载 PROJECT_ROOT/.env，无需外部依赖。"""
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
        # 避免因 .env 格式问题导致服务器启动失败。
        return


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


_load_root_dotenv()

import sys
sys.path.append(str(PROJECT_ROOT))
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
# from retriever import ToolBenchRetriever # 如果从 server 目录运行？
# 如果设置了路径，最好使用相对或绝对路径。
# 由于 app.py 在 dashboard/server 中，如果 CWD 是 dashboard/server 或者 dashboard/server 在路径中，'import retriever' 可以工作。
# 但我们通常从根目录运行。
# 如果我们运行 `python dashboard/server/app.py`，sys.path[0] 是 dashboard/server。所以 `import retriever` 可以工作。
# 但 `agentmark` 需要根目录在路径中。

from agentmark.environments.toolbench.adapter import ToolBenchAdapter
retriever = None
retriever_loading = False

async def init_retriever():
    global retriever, retriever_loading
    retriever_loading = True
    print("[INFO] Background: Initializing ToolBench Retriever on CPU...")
    try:
        # 在线程中运行以避免阻塞简单初始化
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

@app.on_event("startup")
async def startup_event():
    print("[INFO] Initializing ToolBench Retriever...")
    asyncio.create_task(init_retriever())


# --- Simulation State ---

class AgentState:
    """封装单个代理的状态（基线或带水印）"""
    def __init__(self, task_data: Dict, role: str):
        self.role = role # 'baseline' 或 'watermarked'
        self.task = copy.deepcopy(task_data) # 深拷贝以确保独立修改
        
        # ToolBench 适配器状态
        # 对于此演示，我们使用简化的适配器，依赖 LLM 提出 JSON
        self.adapter = ToolBenchAdapter(TOOL_DATA_ROOT)
        self.episode = self.adapter.prepare_episode(self.task)
        
        # 执行历史
        self.trajectory = [] # {role, message} 列表
        self.swarm_history: List[Dict[str, Any]] = []
        self.step_count = 0
        self.last_observation = ""
        self.done = False

class Session:
    def __init__(self, session_id: str, api_key: str, task_data: Dict, payload: str = "1101"):
        self.session_id = session_id
        self.start_time = time.time()
        
        # 通用配置
        self.max_steps = 15
        
        # 代理状态
        self.watermarked_state = AgentState(task_data, 'watermarked')
        self.baseline_state = AgentState(task_data, 'baseline')
        
        # 载荷 / 水印状态（仅用于带水印的代理）
        self.bit_stream_str_raw = payload if payload else "1101" # 保留原始值以供参考
        # 初始化 RLNC
        self.rlnc = DeterministicRLNC(self.bit_stream_str_raw)
        self.bit_index = 0
        
        # LLM 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"
        self.evaluation_result = None # 存储评估结果

sessions: Dict[str, Session] = {}

# --- Swarm Plugin Mode ---

def get_weather(location: str, time: str = "now") -> str:
    """获取给定位置的当前天气。位置必须是城市。"""
    return json.dumps({"location": location, "temperature": "65", "time": time})


def get_weather_forecast(location: str, days: str = "3") -> str:
    """获取给定位置和天数的短期天气预报。"""
    try:
        days_val = int(days)
    except Exception:
        days_val = 3
    return json.dumps(
        {"location": location, "days": days_val, "forecast": ["sunny", "cloudy", "rain"]}
    )


def get_air_quality(location: str) -> str:
    """获取给定位置的简单空气质量报告。"""
    return json.dumps({"location": location, "aqi": 42, "status": "good"})


def send_email(recipient: str, subject: str, body: str) -> str:
    """发送简短邮件。"""
    print("Sending email...")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return "Sent!"


def send_sms(phone_number: str, message: str) -> str:
    """向电话号码发送简短短信。"""
    print("Sending sms...")
    print(f"To: {phone_number}")
    print(f"Message: {message}")
    return "Sent!"

def get_top_rated_movies(limit: int = 10, min_imdb: float = 8.0) -> str:
    """返回带有 IMDb 评分的顶级电影列表。"""
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
    """按类型搜索电影。"""
    return json.dumps(
        {
            "genre": genre,
            "limit": limit,
            "results": ["Inception", "Interstellar", "The Matrix"],
        }
    )


def get_movie_summary(title: str) -> str:
    """获取电影标题的简短摘要。"""
    return json.dumps(
        {
            "title": title,
            "summary": "A brief synopsis for the requested movie.",
        }
    )


def search_web(query: str) -> str:
    """为一般查询搜索网络。"""
    return json.dumps({"query": query, "results": []})


ADD_AGENT_SYSTEM_PROMPT = "You are a helpful agent."
TOOLBENCH_SWARM_PROMPT = (
    "You are a tool-using agent. Use the provided tools to solve the task. "
    "Call a tool when needed; otherwise, respond with the final answer."
)

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

_SWARM_ADD_AGENT = None
_SWARM_TOOLBENCH_AGENT = None


def _get_swarm_add_agent():
    global _SWARM_ADD_AGENT
    if _SWARM_ADD_AGENT is None:
        from swarm import Agent

        _SWARM_ADD_AGENT = Agent(
            name="General Tool Agent",
            instructions=ADD_AGENT_SYSTEM_PROMPT,
            functions=[
                get_weather,
                get_weather_forecast,
                get_air_quality,
                get_top_rated_movies,
                search_movies_by_genre,
                get_movie_summary,
                search_web,
                send_email,
                send_sms,
            ],
        )
    return _SWARM_ADD_AGENT


def _get_swarm_toolbench_agent():
    global _SWARM_TOOLBENCH_AGENT
    if _SWARM_TOOLBENCH_AGENT is None:
        from swarm import Agent

        _SWARM_TOOLBENCH_AGENT = Agent(
            name="ToolBench Agent",
            instructions=TOOLBENCH_SWARM_PROMPT,
        )
    return _SWARM_TOOLBENCH_AGENT


def _toolbench_param_to_schema(param: Any) -> Optional[Dict[str, Any]]:
    if isinstance(param, dict):
        name = param.get("name") or param.get("parameter") or param.get("field")
        if not name:
            return None
        param_type = (param.get("type") or param.get("param_type") or "string").lower()
        description = (param.get("description") or param.get("desc") or "").strip()
    elif isinstance(param, str):
        name = param
        param_type = "string"
        description = ""
    else:
        return None

    if param_type not in {"string", "number", "integer", "boolean", "object", "array"}:
        param_type = "string"

    schema = {"name": name, "type": param_type}
    if description:
        schema["description"] = description
    return schema


def _build_toolbench_tools(tool_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool in tool_summaries:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or ""
        if not name:
            continue
        description = (tool.get("description") or "").strip()
        required_items = tool.get("required_parameters") or []
        optional_items = tool.get("optional_parameters") or []

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for item in required_items:
            schema = _toolbench_param_to_schema(item)
            if not schema:
                continue
            properties[schema["name"]] = {k: v for k, v in schema.items() if k != "name"}
            required.append(schema["name"])

        for item in optional_items:
            schema = _toolbench_param_to_schema(item)
            if not schema:
                continue
            if schema["name"] in properties:
                continue
            properties[schema["name"]] = {k: v for k, v in schema.items() if k != "name"}

        parameters: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return tools


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
        user_lines = "候选动作：\n" + "\n".join(f"- {c}" for c in candidates)
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
            tool_lines = "\n可用工具参数：\n" + "\n".join(tool_specs)
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] = (msg["content"] or "") + "\n" + user_lines + tool_lines
                break
        else:
            messages.append({"role": "user", "content": user_lines + tool_lines})

    injected = [{"role": "system", "content": instr}]
    injected.extend(messages)
    injected[0]["content"] += "\n[AgentMark mode=tools]"
    return injected


def _render_prompt_messages(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        if content is None:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


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


_PROXY_BASE_CACHE: Optional[str] = None
_BASE_LLM_BASE_CACHE: Optional[str] = None

_BASE_MODEL_MAP = {
    "gpt-4o": "deepseek-chat",
    "gpt-4o-mini": "deepseek-chat",
    "gpt-4-turbo": "deepseek-chat",
    "gpt-4": "deepseek-chat",
    "gpt-3.5-turbo": "deepseek-chat",
}


def _get_proxy_base() -> str:
    global _PROXY_BASE_CACHE
    if _PROXY_BASE_CACHE is None:
        _PROXY_BASE_CACHE = (
            os.getenv("AGENTMARK_PROXY_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:8001/v1"
        )
        print(f"[INFO] AgentMark proxy base: {_PROXY_BASE_CACHE}")
    return _PROXY_BASE_CACHE


def _get_base_llm_base() -> str:
    global _BASE_LLM_BASE_CACHE
    if _BASE_LLM_BASE_CACHE is None:
        _BASE_LLM_BASE_CACHE = (
            os.getenv("AGENTMARK_BASE_LLM_BASE")
            or os.getenv("TARGET_LLM_BASE")
            or "https://api.deepseek.com"
        )
        print(f"[INFO] AgentMark base LLM base: {_BASE_LLM_BASE_CACHE}")
    return _BASE_LLM_BASE_CACHE


def _resolve_base_model(requested_model: str) -> str:
    override = os.getenv("AGENTMARK_BASE_MODEL") or os.getenv("TARGET_LLM_MODEL")
    if override:
        return override
    model_map_env = os.getenv("TARGET_LLM_MODEL_MAP")
    if model_map_env:
        try:
            model_map = json.loads(model_map_env)
            return model_map.get(requested_model, requested_model)
        except json.JSONDecodeError:
            pass
    return _BASE_MODEL_MAP.get(requested_model, requested_model)


def _build_proxy_client(api_key: Optional[str]) -> OpenAI:
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY") or "anything",
        base_url=_get_proxy_base(),
    )


def _build_base_llm_client(api_key: Optional[str]) -> OpenAI:
    return OpenAI(
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "anything",
        base_url=_get_base_llm_base(),
    )


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


class AddAgentInitRequest(BaseModel):
    apiKey: Optional[str] = None
    repoUrl: Optional[str] = ""


class AddAgentTurnRequest(BaseModel):
    sessionId: str
    message: str
    apiKey: Optional[str] = None


class AddAgentEvaluateRequest(BaseModel):
    sessionId: str
    language: Optional[str] = "en"

# --- Helpers ---
def build_messages(query: str, tool_summaries: List[str], admissible_commands: List[str]) -> List[Dict]:
    # 构建与 ToolBench 兼容的系统提示
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
    # DeepSeek chat API 在此演示中不暴露 logprobs。
    # 因此我们：
    #   1) 如果存在，优先使用模型提供的 `action_weights`（已归一化）。
    #   2) 否则，回退到有偏但非退化的分布（多个"步骤"），
    #      以避免"一个巨大 + 许多相同微小"的形状，这种形状会在 UI 中折叠桶。

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
        # 有偏但非退化的分布：top_action 获得固定质量，
        # 其余获得几何衰减，使得 p2>p3>...（无平坦平台）。
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
    
    # 加载场景数据
    # 在实际应用中，这将从磁盘/数据库加载。
    # 如果场景查询是自定义的，我们将使用 'retriever' 来查找工具？
    # 对于固定场景，我们可能已经有了工具列表。
    # 让我们假设我们总是为"实时"演示动态检索，或使用缓存。
    
    task = {
        "query": "Solve task " + req.scenarioId, 
        "api_list": [], # 将为空，适配器处理回退或我们检索
        "id": req.scenarioId,
        "payload_str": req.payload 
    }
    
    # 如果我们知道查询，尝试检索真实工具？
    # 目前，从空或基本开始。
    
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
        "totalSteps": 0, # 开始
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


@app.post("/api/add_agent/start")
async def start_add_agent_session(req: AddAgentInitRequest):
    session_id = f"agent_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    add_agent_sessions[session_id] = AddAgentSession(
        session_id=session_id,
        api_key=_resolve_api_key(req.apiKey),
        repo_url=req.repoUrl or "",
    )
    return {
        "sessionId": session_id,
        "proxyBase": _get_proxy_base(),
    }


@app.post("/api/add_agent/turn")
async def add_agent_turn(req: AddAgentTurnRequest):
    if req.sessionId not in add_agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")
    session = add_agent_sessions[req.sessionId]
    model_name = os.getenv("AGENTMARK_TARGET_MODEL", "gpt-4o")
    completion = None
    started_at = time.time()
    use_swarm = (os.getenv("AGENTMARK_USE_SWARM") or "1").strip().lower() not in {"0", "false", "no"}
    use_baseline = (os.getenv("AGENTMARK_ADD_AGENT_BASELINE") or "1").strip().lower() not in {"0", "false", "no"}
    proxy_base = _get_proxy_base()
    agentmark_body = {
        "session_id": session.session_id,
        "context": f"{session.session_id}||step{session.step_count}",
        "use_scoring_tool": True,
    }
    messages = [
        {"role": "system", "content": ADD_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": req.message.strip()},
    ]

    # 定义同步任务以并行执行
    def run_watermarked_model():
        w_start = time.time()
        w_completion = None
        
        if use_swarm:
            try:
                from swarm import Swarm
                # 使用 run_and_stream 以获得真实流程（即使工具在此处不可执行）
                # 注意：AddAgent 模式通常只预测*下一步*（评分）。
                # 但如果我们想支持多轮自回归，我们需要可执行工具。
                # 当前设置可能没有可执行工具，因此它会在 ToolCall 处停止。
                
                swarm_client = Swarm(client=_build_proxy_client(req.apiKey or session.api_key))
                
                # 我们需要构造一个模拟代理对象
                from swarm import Agent
                ephemeral_agent = Agent(name="AddAgent", model=model_name, instructions=ADD_AGENT_SYSTEM_PROMPT)
                
                # 标准 Swarm 运行（如果工具可执行则处理循环，否则一轮）
                # 由于我们没有 ADD_AGENT_TOOLS 的 Python 函数，我们只将它们作为 tools_schema 传递？
                # Swarm 需要可调用对象的 'functions' 列表用于 execute_tools=True。
                # 如果我们传递 'tools_override'，Swarm 将其用于 API 调用模式。
                # 但如果 'functions' 为空且 'execute_tools=True'，执行将失败。
                # 所以我们必须设置 execute_tools=False 以防止 Swarm 尝试调用不存在的函数。
                # 这意味着它的行为就像 get_chat_completion（单轮）。
                # 这确认了"Add Agent"充当预测器，而不是运行器。
                # 如果需要，我将保留此行为但使用 `run` 接口以保持一致性，
                # 或者意识到对于"Add Agent"，单轮确实是此工具的真实流程。
                
                # 但是，为了满足"真实 Swarm 流程"，我将使用 `run` 但设置 `execute_tools=False`。
                # 这确保使用 Swarm 的内部提示/消息逻辑。
                
                response = swarm_client.run(
                    agent=ephemeral_agent,
                    messages=[{"role": "user", "content": req.message.strip()}],
                    context_variables={},
                    model_override=model_name,
                    stream=False,
                    debug=True,
                    # extra_body={"agentmark": agentmark_body}, # Swarm run 不支持 extra_body
                    tools_override=ADD_AGENT_TOOLS,
                    execute_tools=False
                )
                
                # 包装响应以匹配下游期望的 ChatCompletion 接口
                class MockChoice:
                    def __init__(self, msg): self.message = msg
                class MockCompletion:
                    def __init__(self, msg): self.choices = [MockChoice(msg)]
                    
                w_completion = MockCompletion(response.messages[-1])
                
            except Exception as exc:
                print(f"[WARN] Swarm bridge failed, falling back to direct call: {exc}")

        if w_completion is None:
            client = _build_proxy_client(req.apiKey or session.api_key)
            try:
                w_completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=ADD_AGENT_TOOLS,
                    extra_body={"agentmark": agentmark_body},
                    temperature=0.0,
                )
            except Exception as exc:
                # 抛出异常以被 asyncio gather 捕获
                raise exc
        
        w_latency = time.time() - w_start
        return w_completion, w_latency

    def run_baseline_model():
        b_step = None
        b_trace = None
        b_start = time.time()
        try:
            base_client = _build_base_llm_client(req.apiKey or session.api_key)
            base_messages = _build_add_agent_scoring_messages(req.message.strip())
            b_trace = {
                "scoring_messages": base_messages,
                "scoring_prompt_text": _render_prompt_messages(base_messages),
            }
            base_completion = base_client.chat.completions.create(
                model=_resolve_base_model(model_name),
                messages=base_messages,
                temperature=0.0,
            )
            base_latency = time.time() - b_start
            b_step = _build_baseline_step(
                base_completion,
                base_latency,
                candidates=_get_add_agent_candidates(),
            )
        except Exception as exc:
            print(f"[WARN] Baseline call failed: {exc}")
        
        return b_step, b_trace

    # 并行执行
    futures = [asyncio.to_thread(run_watermarked_model)]
    if use_baseline:
        futures.append(asyncio.to_thread(run_baseline_model))
    
    results = await asyncio.gather(*futures, return_exceptions=True)

    # 处理带水印的结果
    if isinstance(results[0], Exception):
        exc = results[0]
        # 如果是代理连接问题，严格重新抛出为 502，否则为通用 500
        detail = f"Proxy call failed ({proxy_base}): {exc}"
        raise HTTPException(status_code=502, detail=detail)
    
    completion, latency = results[0]

    # 处理基线结果
    baseline_step = None
    baseline_prompt_trace = None
    if use_baseline:
        if isinstance(results[1], Exception):
            print(f"[WARN] Baseline task experienced unexpected error: {results[1]}")
        else:
            baseline_step, baseline_prompt_trace = results[1]
    watermark = _extract_watermark(completion)
    if not watermark:
        raise HTTPException(status_code=500, detail="Missing watermark in response")
    completion_message = None
    try:
        if completion and completion.choices:
            completion_message = completion.choices[0].message
    except Exception:
        completion_message = None
    completion_content = ""
    completion_tool_calls: Optional[List[Dict[str, Any]]] = None
    if completion_message is not None:
        completion_content = completion_message.content or ""
        tool_calls = _extract_tool_calls(completion_message)
        if tool_calls:
            completion_tool_calls = tool_calls
    tokens_used = _extract_tokens_used(completion)

    prompt_trace = watermark.get("prompt_trace") or {}
    if tokens_used <= 0:
        scoring_text = prompt_trace.get("scoring_prompt_text") or ""
        exec_text = prompt_trace.get("execution_prompt_text") or ""
        approx = (len(scoring_text) + len(exec_text)) / 4.0
        if approx > 0:
            tokens_used = max(1.0, approx)

    if latency <= 0:
        latency = 0.001

    step = _build_add_agent_step(
        watermark,
        session.step_count,
        completion_content,
        completion_tool_calls,
        latency=latency,
        tokens=tokens_used,
    )
    if baseline_step:
        step["baseline"] = baseline_step
    user_msg = req.message.strip()
    session.last_user_message = user_msg
    if not session.task_query:
        session.task_query = user_msg

    session.watermarked_trajectory.append({"role": "user", "message": user_msg})
    session.baseline_trajectory.append({"role": "user", "message": user_msg})

    wm_action = watermark.get("action") or ""
    wm_args = watermark.get("action_args") or {}
    wm_thought = step.get("thought") or ""
    wm_payload: Dict[str, Any] = {"action": wm_action, "thought": wm_thought}
    if wm_args:
        wm_payload["action_args"] = wm_args
    session.watermarked_trajectory.append(
        {"role": "assistant", "message": json.dumps(wm_payload, ensure_ascii=False)}
    )

    base_payload: Dict[str, Any] = {}
    if baseline_step:
        base_action = (baseline_step.get("action") or "").strip()
        base_action_name = base_action.replace("Call: ", "").strip() if base_action.startswith("Call: ") else base_action
        base_payload = {"action": base_action_name, "thought": baseline_step.get("thought") or ""}
        if base_action_name == "Finish" and baseline_step.get("finalAnswer"):
            base_payload["action_args"] = {"final_answer": baseline_step.get("finalAnswer")}
    else:
        base_payload = wm_payload
    session.baseline_trajectory.append(
        {"role": "assistant", "message": json.dumps(base_payload, ensure_ascii=False)}
    )
    session.step_count += 1
    return {
        "sessionId": req.sessionId,
        "step": step,
        "promptTrace": prompt_trace,
        "baselinePromptTrace": baseline_prompt_trace,
        "watermark": watermark,
    }


@app.post("/api/add_agent/evaluate")
async def evaluate_add_agent(req: AddAgentEvaluateRequest):
    if req.sessionId not in add_agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = add_agent_sessions[req.sessionId]
    if not sess.watermarked_trajectory or not sess.baseline_trajectory:
        raise HTTPException(status_code=400, detail="Not enough data to evaluate")

    lang_instruction = "Reasoning must be in English."
    if req.language == "zh":
        lang_instruction = "请使用中文进行简要评价 (Reasoning must be in Chinese)."

    def summarize_trajectory(traj):
        summary = ""
        for t in traj:
            role = t.get("role")
            msg = t.get("message", "")
            if role == "user":
                summary += f"User: {msg}\n"
            elif role == "assistant":
                try:
                    data = json.loads(msg)
                    summary += f"Assistant Thought: {data.get('thought')}\nAssistant Action: {data.get('action')}\n"
                    if "final_answer" in data.get("action_args", {}):
                        summary += f"Assistant Final Answer: {data['action_args']['final_answer']}\n"
                except Exception:
                    summary += f"Assistant: {msg}\n"
            elif role == "tool":
                summary += f"Tool Output: {msg[:200]}...\n"
        return summary

    baseline_summary = summarize_trajectory(sess.baseline_trajectory)
    watermarked_summary = summarize_trajectory(sess.watermarked_trajectory)
    query = sess.task_query or sess.last_user_message or "Add agent task"

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
        content = completion.choices[0].message.content or ""
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Invalid evaluation payload")
        json_str = content[start:end+1]
        res_json = json.loads(json_str)

        swapped = not is_baseline_A
        if swapped:
            final_result = {
                "model_a_score": res_json.get("model_b_score", 0),
                "model_b_score": res_json.get("model_a_score", 0),
                "reason": res_json.get("reason", "")
            }
        else:
            final_result = res_json

        sess.evaluation_result = final_result
        return final_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add_agent/{session_id}/save")
async def save_add_agent_session(session_id: str):
    """将 Add Agent 会话保存到历史记录"""
    if session_id not in add_agent_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        sess = add_agent_sessions[session_id]
        
        # 将带水印和基线合并为统一的历史视图
        # 对于简单的历史查看，我们可以只保存 UI 中使用的 'steps' 结构
        # 我们需要构造类似于 AddAgentDashboard 维护的 'steps' 列表
        
        # 实际上，AddAgentDashboard 维护 'steps' 状态。
        # 但我们在这里只有会话中的原始轨迹。
        # 然而，sess.watermarked_trajectory 包含结构化的思考/动作。
        
        # 更好的方法：前端已经有了完整的 'steps' 状态数组。
        # 最好让前端使用它拥有的数据调用 'save_scenario'，
        # 指定 type='add_agent'。
        # 如果前端发送数据，我们不需要特殊的后端重构逻辑。
        
        # 因此，我们将跳过这里的复杂逻辑，只依赖前端使用通用保存 API。
        pass
    except Exception as e:
        print(f"[ERROR] Save add agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Scenario Persistence ---

@app.get("/api/scenarios")
async def list_scenarios(search: Optional[str] = None, limit: int = 100, type: Optional[str] = None):
    """列出数据库中所有保存的对话，支持可选的搜索和类型过滤"""
    try:
        scenarios = db.list_conversations(limit=limit, search=search, type_filter=type)
        return scenarios
    except Exception as e:
        print(f"[ERROR] Failed to list scenarios: {e}")
        return []

class SaveScenarioRequest(BaseModel):
    title: Any # str or dict
    data: Dict
    id: Optional[str] = None # 可选的 ID 以覆盖
    type: Optional[str] = "benchmark" # 默认为 benchmark

@app.post("/api/save_scenario")
async def save_scenario(req: SaveScenarioRequest):
    """将对话保存到数据库"""
    try:
        scenario_id = req.id if req.id else str(uuid.uuid4())
        
        scenario_data = req.data
        scenario_data["id"] = scenario_id
        scenario_data["type"] = req.type
        
        # 处理标题格式
        if isinstance(req.title, str):
            scenario_data["title"] = {"en": req.title, "zh": req.title}
        else:
            scenario_data["title"] = req.title
        
        # 保存到数据库
        db.save_conversation(scenario_data)
        
        print(f"[INFO] Saved scenario {scenario_id} to database")
        return {"status": "success", "id": scenario_id}
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/scenarios/clear_all")
async def clear_all_history():
    """清除数据库中的所有对话历史"""
    try:
        deleted_count = db.clear_all_conversations()
        print(f"[INFO] Cleared all history: {deleted_count} conversations deleted")
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        print(f"[ERROR] Clear all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scenarios/batch_delete")
async def batch_delete_scenarios(req: dict):
    """批量删除多个场景"""
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
    """切换对话的置顶状态"""
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
    """从数据库中删除对话"""
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
    history: List[Dict] # {role, content/message} 列表

@app.post("/api/generate_title")
async def generate_title(req: GenerateTitleRequest):
    try:
        # 提取用户消息以进行摘要
        # 限制为前几轮以生成标题
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

        # 使用快速调用来生成标题
        # 我们可以使用相同的客户端
        # 如果需要，创建临时客户端，或重用会话中的客户端（如果可用）？
        # 我们这里不一定有会话 ID。
        # 但我们用键初始化了 'sessions'。我们可以只实例化一个通用客户端。
        # 但是，为了避免不必要的全局客户端初始化，我们可以选择一个活动会话或初始化一个临时会话。
        # 或者更好：为实用任务初始化全局客户端。
        
        # 注意：在此演示中，我们假设我们有 API 密钥。
        # 但此请求来自前端。它有 API 密钥吗？
        # 如果是"自动保存"，前端可能不会在这里传递 API 密钥。
        # 理想情况下，我们应该在请求中传递 API 密钥或重用全局环境变量。
        # 对于此演示，让我们假设我们重用任何活动会话或环境中的有效 API 密钥。
        # 如果没有活动会话，我们可能会失败。
        # 让我们检查是否有任何活动会话可以获取凭据，或使用配置的默认值。
        
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
    """从数据库恢复会话"""
    print(f"[INFO] Restore session request for scenarioId: {req.scenarioId}")
    
    # 1. 从数据库加载保存的场景
    data = db.get_conversation(req.scenarioId)
    
    if not data:
        print(f"[ERROR] Scenario {req.scenarioId} not found in database")
        # 列出所有可用对话以供调试
        all_convs = db.list_conversations(limit=10)
        print(f"[INFO] Available conversations: {[c['id'] for c in all_convs]}")
        raise HTTPException(status_code=404, detail="Saved scenario not found")
    
    print(f"[INFO] Found scenario in database: {data.get('id')}, steps: {len(data.get('steps', []))}")

    # 2. 初始化会话
    session_id = f"sess_{int(time.time())}_{req.scenarioId}_restored"
    
    # 从保存的数据中提取任务详情
    # 保存的数据有 'steps'、'userQuery' 等
    task = {
        "query": data.get("userQuery") or "Restored Task",
        "api_list": [], # 我们将依赖检索进行下一步或假设无状态
        "id": req.scenarioId,
        "payload_str": data.get("payload") or "11001101"
    }
    
    # 3. 创建会话
    api_key = _resolve_api_key(req.apiKey)
    session = Session(session_id, api_key, task, task["payload_str"])
    
    # 4. 从步骤重构轨迹
    # 这是从 UI 步骤到内部轨迹的"尽力而为"映射
    # UI 步骤类型：'user_input'、'thought'（带动作/工具）、'tool'、'finish'
    
    watermarked_trajectory = []
    baseline_trajectory = []
    
    steps = data.get("steps", [])
    
    # 我们需要将步骤映射到（用户、助手、工具）消息。
    # 逻辑：
    # - 如果 stepType == 'user_input'：-> 用户消息
    # - 如果 stepType == 'thought' 或 'finish'：-> 助手消息（重构 JSON）
    # - 如果 stepType == 'tool'：-> 工具消息（观察）
    
    # 让我们迭代并重构
    for step in steps:
        s_type = step.get("stepType", "thought")
        
        if s_type == "user_input":
            # 用户消息对两个代理都相同
            user_msg = {"role": "user", "message": step.get("thought") or step.get("action")}
            watermarked_trajectory.append(user_msg)
            baseline_trajectory.append(user_msg)
            
        elif s_type in ["thought", "finish", "tool"]:
            # 重构带水印代理的消息
            thought = step.get("thought", "")
            action = step.get("action", "")
            final_answer = step.get("finalAnswer")
            
            # 辅助函数解析 "Call: ToolName" -> ToolName
            chosen_tool = "Finish"
            if action.startswith("Call: "):
                chosen_tool = action.replace("Call: ", "").strip()
            elif action == "Finish":
                chosen_tool = "Finish"
            
            # 为带水印代理重构字典
            model_out_dict = {
                "action": chosen_tool,
                "action_args": {},
                "thought": thought
            }
            
            if chosen_tool == "Finish" and final_answer:
                 model_out_dict["action_args"] = { "final_answer": final_answer }
            
            # 存储为字符串（模拟 LLM 原始输出）
            watermarked_trajectory.append({"role": "assistant", "message": json.dumps(model_out_dict)})
            
            # 为带水印代理添加观察
            obs = step.get("toolDetails") or step.get("observation")
            if obs and chosen_tool != "Finish":
                watermarked_trajectory.append({"role": "tool", "message": obs})
            
            # 重构基线代理的消息（如果存在）
            baseline_data = step.get("baseline")
            if baseline_data:
                baseline_thought = baseline_data.get("thought", "")
                baseline_action = baseline_data.get("action", "")
                baseline_final_answer = baseline_data.get("finalAnswer")
                
                # 解析基线动作
                baseline_tool = "Finish"
                if baseline_action.startswith("Call: "):
                    baseline_tool = baseline_action.replace("Call: ", "").strip()
                elif baseline_action == "Finish":
                    baseline_tool = "Finish"
                
                # 为基线代理重构字典
                baseline_model_dict = {
                    "action": baseline_tool,
                    "action_args": {},
                    "thought": baseline_thought
                }
                
                if baseline_tool == "Finish" and baseline_final_answer:
                    baseline_model_dict["action_args"] = { "final_answer": baseline_final_answer }
                
                baseline_trajectory.append({"role": "assistant", "message": json.dumps(baseline_model_dict)})
                
                # 为基线代理添加观察
                baseline_obs = baseline_data.get("toolDetails") or baseline_data.get("observation")
                if baseline_obs and baseline_tool != "Finish":
                    baseline_trajectory.append({"role": "tool", "message": baseline_obs})
            else:
                # 如果没有基线数据，复制带水印的数据
                baseline_trajectory.append({"role": "assistant", "message": json.dumps(model_out_dict)})
                if obs and chosen_tool != "Finish":
                    baseline_trajectory.append({"role": "tool", "message": obs})

    # 用各自的轨迹填充两个代理
    session.watermarked_state.trajectory = watermarked_trajectory
    session.baseline_state.trajectory = baseline_trajectory
    
    # 设置步数
    session.watermarked_state.step_count = len(steps)
    session.baseline_state.step_count = len(steps)
    
    # 存储会话
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
    
    # 为继续提示检索新工具
    print(f"\n[INFO] >>> RECEIVED CONTINUE PROMPT: '{req.prompt}' <<<\n")
    if retriever:
        new_tools = retriever.retrieve(req.prompt, top_k=5)
        if new_tools:
            print(f"[INFO] Retrieved {len(new_tools)} new tools for continuation.")
            
            # 辅助函数用新工具更新代理状态
            def update_agent_tools(agent_state: AgentState):
                # 基本去重检查
                current_tools = agent_state.task.get("api_list", [])
                existing_names = {t.get("func_name") or t.get("api_name") for t in current_tools}
                
                for tool in new_tools:
                    t_name = tool.get("func_name") or tool.get("api_name")
                    if t_name not in existing_names:
                        current_tools.append(tool)
                        existing_names.add(t_name)
                
                agent_state.task["api_list"] = current_tools
                
                # 关键：重新初始化 episode 以刷新工具摘要和可接受命令
                try:
                    updated_episode = agent_state.adapter.prepare_episode(agent_state.task)
                    agent_state.episode["tool_summaries"] = updated_episode["tool_summaries"]
                    agent_state.episode["admissible_commands"] = updated_episode["admissible_commands"]
                except Exception as e:
                    print(f"[ERROR] Failed to refresh episode context for {agent_state.role}: {e}")

            # 更新两个代理
            update_agent_tools(sess.watermarked_state)
            update_agent_tools(sess.baseline_state)
            print("[INFO] Updated tools for both agents.")
            
    
    # 为两者追加用户提示到轨迹
    sess.watermarked_state.trajectory.append({"role": "user", "message": req.prompt})
    sess.baseline_state.trajectory.append({"role": "user", "message": req.prompt})
    
    # 扩展最大步数以允许继续
    sess.max_steps += 10
    
    # 关键：重置完成状态以便代理继续
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
        # 如果完成，立即返回 JSON 以保持一致性
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
        
        # 检查完成状态
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
                # Swarm 原生执行（为同步客户端线程化）
                
                # 1. 定义动态工具包装器工厂
                def make_wrapper(tool_def, adapter, task_state, queue, loop, agent_role):
                    tool_name = tool_def["name"]
                    
                    def wrapper(**kwargs):
                        # 副作用：通知 UI 工具调用开始
                        # 这在 Swarm 循环内同步发生（阻塞线程，而不是主循环）
                        # 我们使用 call_soon_threadsafe 推送到异步队列
                        
                        tool_display = f"Call: {tool_name}"
                        call_details = json.dumps(kwargs, ensure_ascii=False)
                        
                        # 通知 UI：工具调用意图（作为思考或特殊事件）
                        # 由于 Swarm 在流中直到之后才产生"我正在调用 X"？
                        # 实际上 Swarm 流式传输内容，然后执行工具。
                        # 我们注入特定事件以立即显示工具调用卡片。
                        
                        loop.call_soon_threadsafe(
                             queue.put_nowait,
                             {
                                 "type": "thought",
                                 "agent": agent_role,
                                 "content": f"\n\n[Action: {tool_name}]\n"
                             }
                        )

                        action_obj = {"tool": tool_name, "arguments": kwargs}
                        
                        # Execute logic using adapter (preserves cache/fake)
                        obs_result = adapter.step(
                            action_obj,
                            agent_state.episode["tool_summaries"],
                            state=task_state
                        )
                        return obs_result["observation"]

                    wrapper.__name__ = tool_name
                    wrapper.__doc__ = tool_def.get("description", "")
                    return wrapper

                # 2. Setup Tools and Agent
                from swarm import Swarm, Agent
                
                tool_summaries = agent_state.episode["tool_summaries"]
                tools_override = _build_toolbench_tools(tool_summaries)
                
                # specific loop for threadsafe calls
                loop = asyncio.get_running_loop()
                
                funcs = []
                func_map = {}
                for t in tool_summaries:
                    w = make_wrapper(t, agent_state.adapter, agent_state.task, output_queue, loop, agent_state.role)
                    funcs.append(w)
                    func_map[t["name"]] = w

                # Swarm Agent Definition
                # Note: We use the simple prompt as requested
                swarm_agent = Agent(
                    name="Swarm Assistant",
                    instructions="You are a helpful agent. Use the provided tools to solve the task. Call a tool when needed; otherwise, respond with the final answer.",
                    functions=funcs 
                )
                
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
