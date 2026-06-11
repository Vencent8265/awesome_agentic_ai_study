"""
Stage 3 练习5：错误处理
- 用 Anthropic SDK 格式连接 DeepSeek v4 pro
- 核心概念：tool error 是 data，不是 exception
- LLM 自己决定 retry / 改 query / 放弃
"""

import json
import random
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/anthropic",
)

MODEL = "deepseek-v4-pro"

# ── Tool Schema（给 LLM 看的说明书）──────────────────────────────────────────

TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Use when the user asks about current weather in a city. "
            "Returns weather info or an error dict if the network fails."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如 '台北' / 'Tokyo'"},
            },
            "required": ["city"],
        },
    }
]

# ── Tool Implementation（真正执行的函数）─────────────────────────────────────

def get_weather(city: str) -> dict:
    """模拟偶发网络失败：约 50% 概率失败，方便观察 LLM 的 retry 行为。"""
    if random.random() < 0.5:
        # ✅ Good：返回结构化 error dict，不 raise
        # LLM 看到 retry_hint 后可以自己决定要不要重试
        return {
            "error": "network timeout",
            "category": "transient",
            "retry_hint": "Transient error, please try the same city again.",
        }
    # 正常返回
    fake_data = {
        "台北": {"forecast": "rain", "temperature_c": 24},
        "Tokyo": {"forecast": "sunny", "temperature_c": 28},
        "Sydney": {"forecast": "cloudy", "temperature_c": 18},
    }
    result = fake_data.get(city, {"forecast": "partly cloudy", "temperature_c": 22})
    return {"city": city, **result}

TOOL_IMPL = {
    "get_weather": lambda args: get_weather(args["city"]),
}

# ── ReAct Loop ────────────────────────────────────────────────────────────────

def run_agent(user_query: str, max_iter: int = 6) -> str:
    print(f"\n{'='*60}")
    print(f"用户问题：{user_query}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_query}]

    for step in range(max_iter):
        print(f"\n── Step {step + 1} ──────────────────────────")

        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Anthropic 格式：stop_reason 判断是否调用工具
        print(f"stop_reason: {resp.stop_reason}")

        # 把 assistant 回复接回 messages
        messages.append({"role": "assistant", "content": resp.content})

        # 没有 tool_use → LLM 认为任务完成，退出循环
        if resp.stop_reason == "end_turn":
            final = next(
                (b.text for b in resp.content if hasattr(b, "text")), ""
            )
            print(f"\n✅ 最终回答：{final}")
            return final

        # 有 tool_use → 执行工具，把结果还给 LLM
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_args = block.input  # Anthropic：直接是 dict，不需要 json.loads
            print(f"调用工具：{tool_name}，参数：{tool_args}")

            obs = TOOL_IMPL[tool_name](tool_args)
            print(f"工具返回：{obs}")

            # Anthropic 格式：tool_result 放进 user message 的 content list
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(obs, ensure_ascii=False),
            })

        # 把所有 tool_result 一起放进下一轮 user message
        messages.append({"role": "user", "content": tool_results})

    # 超过 max_iter，graceful end
    final = "已达到最大步骤数，任务未能完成。"
    print(f"\n⚠️  {final}")
    return final

# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 多跑几次，观察 LLM 在工具失败时的 retry 行为
    run_agent("台北现在天气怎么样？")