# Stage 3 - 练习4：多步骤推理
# 4个专用工具，强迫模型走完完整链路
import sys, json
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:3b"

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_population",
        "description": "Look up population of a city in 万人. MUST use this, never guess population.",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "divide",
        "description": "Divide two numbers. Use this to get a ratio between two values.",
        "parameters": {"type": "object",
            "properties": {
                "a": {"type": "float", "description": "numerator"},
                "b": {"type": "float", "description": "denominator"}},
            "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "to_percentage",
        "description": "Convert a ratio (e.g. 0.32) to percentage string (e.g. '32%').",
        "parameters": {"type": "object",
            "properties": {"ratio": {"type": "float"}},
            "required": ["ratio"]}}},
    {"type": "function", "function": {
        "name": "round_int",
        "description": "Round a float to the nearest integer.",
        "parameters": {"type": "object",
            "properties": {"x": {"type": "float"}},
            "required": ["x"]}}},
]

TOOL_IMPL = {
    "lookup_population": lambda a: str({"台北":270,"纽约":840,"东京":1400,"上海":2487}.get(a["city"], 0)),
    "divide":            lambda a: str(a["a"] / a["b"]),
    "to_percentage":     lambda a: f"{round(a['ratio']*100)}%",
    "round_int":         lambda a: str(round(a["x"])),
}

def react(user_message: str, max_steps: int = 8):
    print(f"\n{'='*50}")
    print(f"用户：{user_message}")
    print(f"{'='*50}")
    messages = [
        {"role": "system", "content":
         "Use tools for every step. Never compute or guess numbers yourself. "
         "For population ratios: lookup each city → divide → to_percentage → round_int."},
        {"role": "user", "content": user_message}
    ]
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=300, tools=TOOLS, messages=messages,
        )
        msg = resp.choices[0].message
        if resp.choices[0].finish_reason != "tool_calls" or not msg.tool_calls:
            print(f"\n✅ 最终回答（共{step-1}步）：{msg.content or ''}")
            return
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = TOOL_IMPL[name](args)
            print(f"  Step{step} → {name}({args}) = {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    print("⚠️ 达到最大步数")

react("台北人口是纽约人口的百分之几？用整数表示。")