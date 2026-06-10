import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/anthropic"
)

msg = client.messages.create(
    model="deepseek-v4-pro",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "说出你具体是什么模型型号"
        }
    ]
)

print(msg.content[1].text)


