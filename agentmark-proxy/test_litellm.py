
import litellm
import os

# 1. 将 LiteLLM 指向我们的水印代理
litellm.api_base = "http://localhost:8001/v1"
litellm.api_key = "any-key" # 代理处理向上游的认证
# litellm.drop_params = True # 如需要兼容性，可调整

print("--- Calling LiteLLM via Proxy ---")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get recent news for a company",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"}
                },
                "required": ["symbol"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What is the stock price of NVDA and is there any news?"}]

# 使用标准完成调用
response = litellm.completion(
    model="openai/deepseek-chat", # 强制使用 openai 提供程序以使用我们的通用代理
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print("\n--- LiteLLM Response ---")
print(response)

choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    tc = choice.message.tool_calls[0]
    print(f"\n成功: 通过 LiteLLM 接收到带水印的工具调用！")
    print(f"工具: {tc.function.name}")
    print(f"参数: {tc.function.arguments}")
    # 如果存在，检查推理内容
    print(f"内容（推理）: {choice.message.content}")
else:
    print("\n失败: 未接收到工具调用。")
