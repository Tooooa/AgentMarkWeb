
import os
import json
import httpx
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from agentmark.core.watermark_sampler import sample_behavior_differential
from proxy_utils import inject_top_k_prompt, parse_top_k_response, construct_tool_call_response

app = FastAPI(title="AgentMark Universal Proxy")

# 配置
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.deepseek.com")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY") 
WATERMARK_PAYLOAD = os.getenv("WATERMARK_PAYLOAD", "110101")

# 内存中的会话状态（仅用于概念验证。生产环境请使用 Redis）
session_store = {}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    拦截聊天完成请求。
    如果存在工具，应用单次水印。
    否则，透明转发。
    """
    try:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # 1. 检查这是否是工具调用请求
        tools = body.get("tools")
        if not tools:
            # 透明转发
            return await forward_request(body)

        # 2. 单次转换
        # 修改提示词以要求 Top-K 候选
        original_messages = body.get("messages", [])
        modified_messages = inject_top_k_prompt(original_messages, tools, k=3)
        
        # 准备上游请求（强制 JSON 模式以便解析）
        upstream_body = body.copy()
        upstream_body["messages"] = modified_messages
        upstream_body.pop("tools", None) # 移除工具以防止 LLM 自动执行
        upstream_body.pop("tool_choice", None)
        upstream_body["response_format"] = {"type": "json_object"}
        upstream_body["model"] = "deepseek-chat" # 暂时硬编码以确保兼容性
        
        print(f"[Proxy] 向上游转发 Top-K 请求...")
        
        # 3. 调用上游 LLM
        upstream_response = await call_upstream(upstream_body)
        
        # 4. 解析候选
        try:
            candidates, reasoning = parse_top_k_response(upstream_response) # 返回 (列表, 推理)
        except Exception as e:
            print(f"[Proxy] 解析失败: {e}。回退到原始请求。")
            return await forward_request(body)
            
        if not candidates:
            print("[Proxy] 未返回候选。回退到原始请求。")
            return await forward_request(body)

        # 5. 水印采样
        # 识别会话（简单方法：基于最后一条消息内容的哈希）
        last_msg_content = str(original_messages[-1].get("content", ""))
        session_key = str(hash(last_msg_content)) 
        state = session_store.get(session_key, {"bit_index": 0, "round_num": 0})
        
        # 为 SDK 准备分布
        tool_probs = {c["name"]: c["prob"] for c in candidates}
        # 归一化
        total = sum(tool_probs.values())
        if total > 0:
            tool_probs = {k: v/total for k,v in tool_probs.items()}
        
        # 随机性上下文
        context_str = f"cnt:{state['round_num']}|msg:{last_msg_content[:20]}"
        
        print(f"[Proxy] 从以下工具采样: {tool_probs.keys()}")
        
        selected_tool_name, _, bits_embedded, _ = sample_behavior_differential(
            probabilities=tool_probs,
            bit_stream=WATERMARK_PAYLOAD,
            bit_index=state["bit_index"],
            context_for_key=context_str,
            round_num=state["round_num"]
        )
        
        # 更新状态
        state["bit_index"] += bits_embedded
        state["round_num"] += 1
        session_store[session_key] = state
        
        print(f"[Proxy] 已选择: {selected_tool_name} (嵌入: {bits_embedded} 位)")

        # 6. 查找所选候选的详细信息
        selected_candidate = next((c for c in candidates if c["name"] == selected_tool_name), candidates[0])
        
        # 7. 构造响应
        final_response = construct_tool_call_response(
            original_model=body.get("model", "gpt-4"),
            tool_name=selected_candidate["name"],
            tool_args=selected_candidate["args"],
            content=reasoning # 注入推理
        )
        
        return JSONResponse(content=final_response)

    except Exception as e:
        print(f"[Proxy] 严重错误: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

async def forward_request(body: dict):
    """透明地向上游转发请求。"""
    try:
        response = await call_upstream(body) 
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"Upstream failed: {str(e)}"})

async def call_upstream(body: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 注意：DeepSeek API 使用 OpenAI 兼容格式
        # 如果使用 DeepSeek 端点，确保模型名称正确
        if "deepseek" in UPSTREAM_BASE_URL:
             body["model"] = "deepseek-chat"
             
        resp = await client.post(
            f"{UPSTREAM_BASE_URL}/chat/completions",
            json=body,
            headers=headers
        )
        if resp.status_code != 200:
            print(f"[Proxy] Upstream Error: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
