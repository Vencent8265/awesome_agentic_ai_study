import os
import anthropic
import statistics
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/anthropic"
)

PROMPTS = {
    "中文": "用一句话描述一只猫在做什么。",
    "English": "Describe in one sentence what a cat is doing.",
}

N = 5  # 跑5次

for label, prompt in PROMPTS.items():
    output_tokens = []
    for _ in range(N):
        msg = client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=80,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}]
        )
        output_tokens.append(msg.usage.output_tokens)

    print(f"\n[{label}] prompt: {prompt}")
    print(f"  input tokens:  {msg.usage.input_tokens}")
    print(f"  output tokens — min={min(output_tokens)} max={max(output_tokens)} mean={statistics.mean(output_tokens):.1f}")