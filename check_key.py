
from openai import OpenAI
import os

key = "sk-028c7d27014d4feb892e0d05974f6ff4"
base = "https://api.deepseek.com"

client = OpenAI(api_key=key, base_url=base)

try:
    print(f"Testing key against {base}...")
    res = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    print("Success!")
    print(res.choices[0].message.content)
except Exception as e:
    print(f"Failed: {e}")
