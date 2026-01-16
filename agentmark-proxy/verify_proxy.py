
from openai import OpenAI
import os

# 指向我们的本地代理
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="any-key" # 代理处理向上游的认证
)

print("--- Requesting Tool Call via Proxy ---")

tools = [
    {
        "type": "function",
        "function": {
            "name": "flight_search",
            "description": "Search for flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_check",
            "description": "Check weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "I want to go to Paris. Can you check flights and weather?"}
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools
)

print("\n--- Response Received ---")
print(response)

# 检查是否看起来像标准的工具调用
choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    tc = choice.message.tool_calls[0]
    print(f"\n成功: 通过代理接收到工具调用！")
    print(f"工具: {tc.function.name}")
    print(f"参数: {tc.function.arguments}")
    print(f"推理: {choice.message.content}")
else:
    print("\n失败: 未接收到工具调用。")
