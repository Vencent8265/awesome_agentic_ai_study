# Stage 3 - 练习2：多工具选择
# 运行：python3 practice_2.py

import sys, json
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:3b"

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "SUse for ANY question about real-world facts, events, or people — even if you think you know the answer.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate arithmetic expressions with +, -, *, /, parentheses. Use only for math calculations.",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression, e.g. (19 * 42) - 8"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "calendar_lookup",
        "description": "Look up events or appointments for a specific date. Use only when a specific date is mentioned.",
        "parameters": {"type": "object",
            "properties": {"date": {"type": "string", "description": "Date in YYYY-MM-DD format"}},
            "required": ["date"]}}},
]

# 测试用例：问题 → 预期调用的工具
TEST_CASES = [
    ("What is (19 * 42) - 8?",              "calculator"),
    ("What events do I have on 2024-12-25?", "calendar_lookup"),
    ("Who won the 2024 US election?",        "web_search"),
]

def get_tool_choice(question: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=200,
        tools=TOOLS,
        #强制调用,有效，但失去了"模型自主判断"的意义
        tool_choice={"type": "function", "function": {"name": "web_search"}},
        messages=[{"role": "user", "content": "Who won the 2024 US election?"}],
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        print(f"✓ 调用了：{msg.tool_calls[0].function.name}")
        print(f"  参数：{msg.tool_calls[0].function.arguments}")
    else:
        print("✗ 还是没调用")

print("=" * 50)
correct = 0
for question, expected in TEST_CASES:
    chosen = get_tool_choice(question)
    ok = chosen == expected
    correct += ok
    print(f"{'✓' if ok else '✗'} 问题：{question}")
    print(f"  预期：{expected} | 实际：{chosen}\n")

print("=" * 50)
print(f"结果：{correct}/{len(TEST_CASES)}")
print()
print("💡 观察重点：")
print("   - description 写'何时用'比写'做什么'更重要")
print("   - 三个工具的边界要互斥，否则小模型容易选错")
print("   - qwen2.5:3b 对 description 质量比 Claude 敏感得多")