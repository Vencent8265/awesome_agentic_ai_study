"""
Stage 3 练习6：Function Schema 设计（坏 schema 修到好）
- 同一个工具（温度转换），两种 schema 写法
- 用 DeepSeek v4 pro + Anthropic SDK 格式
- 观察 4 个 schema 改进点对 LLM 行为的影响
"""

import json
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/anthropic",
)

MODEL = "deepseek-v4-pro"

# ── Tool Implementation（执行不变，只有 schema 不同）────────────────────────

def convert_temperature(value: float, unit: str) -> dict:
    """把温度从一个单位转换到另一个单位。"""
    if unit == "celsius":
        result = value * 9 / 5 + 32
        return {"input": f"{value}°C", "output": f"{result:.1f}°F"}
    elif unit == "fahrenheit":
        result = (value - 32) * 5 / 9
        return {"input": f"{value}°F", "output": f"{result:.1f}°C"}
    else:
        return {"error": f"unknown unit: {unit}", "retry_hint": "unit must be 'celsius' or 'fahrenheit'"}

# ── ❌ BAD Schema ─────────────────────────────────────────────────────────────
# 4 个问题：
# 1. name 太笼统（"convert"）
# 2. description 没写"何时用"，只写"做什么"
# 3. type 全是 string
# 4. 没有 required，没有 enum

BAD_TOOL = {
    "name": "convert",
    "description": "Convert a value.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "unit": {"type": "string"},
        },
    },
}

# ── ✅ GOOD Schema ────────────────────────────────────────────────────────────
# 4 个改进：
# 1. name 具体（"convert_temperature"）
# 2. description 写"何时用 + 边界"
# 3. value 改 number，unit 加 enum 收敛
# 4. 明确 required

GOOD_TOOL = {
    "name": "convert_temperature",
    "description": (
        "Use when the user asks to convert a temperature between Celsius and Fahrenheit. "
        "Do NOT use for other unit conversions (weight, distance, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The temperature value to convert.",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of the input value.",
            },
        },
        "required": ["value", "unit"],
    },
}

# ── 执行函数 ──────────────────────────────────────────────────────────────────

def run_with_schema(schema: dict, label: str, query: str):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"用户问题：{query}")
    print(f"{'='*60}")

    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        tools=[schema],
        messages=[{"role": "user", "content": query}],
    )

    print(f"stop_reason: {resp.stop_reason}")

    tool_calls = [b for b in resp.content if b.type == "tool_use"]

    if not tool_calls:
        # LLM 没有调用工具，直接回答了
        text = next((b.text for b in resp.content if hasattr(b, "text")), "")
        print(f"⚠️  LLM 没有调用工具，直接回答：{text[:100]}")
        return

    tc = tool_calls[0]
    print(f"调用工具：{tc.name}")
    print(f"传入参数：{tc.input}")  # Anthropic：直接是 dict

    # 执行工具
    obs = convert_temperature(
        value=float(tc.input.get("value", 0)),
        unit=str(tc.input.get("unit", "")).lower(),
    )
    print(f"工具返回：{obs}")

    # 把结果还给 LLM
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": resp.content},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tc.id, "content": json.dumps(obs, ensure_ascii=False)}
        ]},
    ]

    final_resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        tools=[schema],
        messages=messages,
    )

    final_text = next((b.text for b in final_resp.content if hasattr(b, "text")), "")
    print(f"✅ 最终回答：{final_text}")

# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "100°C 是几度华氏？"

    # 先跑坏的，观察 LLM 传参是否混乱
    run_with_schema(BAD_TOOL, "❌ BAD SCHEMA", query)

    # 再跑好的，对比行为差异
    run_with_schema(GOOD_TOOL, "✅ GOOD SCHEMA", query)

    print("\n" + "="*60)
    print("对比总结：")
    print("BAD  → name 模糊 / description 无场景 / type=string / 无 enum")
    print("GOOD → name 具体 / 写何时用 / type=number / enum 收敛 unit")
    print("="*60)