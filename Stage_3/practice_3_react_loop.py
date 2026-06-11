# Stage 3 - 练习3 修复版
import sys, json
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:3b"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_population",
        "description": "Get the population of a city. MUST use this for any city population data. Never use your own knowledge for population numbers.",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate arithmetic expressions. MUST use this for all calculations.",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}},
]

def get_population(city: str) -> str:
    data = {"台北": 270, "纽约": 840, "东京": 1400, "上海": 2487}
    return f"{data.get(city, 0)}万人" if city in data else f"找不到{city}的数据"

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"

TOOL_IMPL = {
    "get_population": lambda args: get_population(args["city"]),
    "calculator":     lambda args: calculator(args["expression"]),
}

def react(user_message: str, max_steps: int = 6):
    print(f"\n{'='*50}")
    print(f"用户：{user_message}")
    print(f"{'='*50}")

    messages = [
        {"role": "system", "content": 
         "You must use tools for ALL data and calculations. "
         "Never use your own knowledge for population numbers or math. "
         "For population comparisons: first call get_population for EACH city separately, then use calculator."},
        {"role": "user", "content": user_message}
    ]

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            tools=TOOLS,
            messages=messages,
        )
        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if finish != "tool_calls" or not msg.tool_calls:
            text = msg.content or ""
            print(f"\n✅ 最终回答（共{step-1}步）：{text}")
            return

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = TOOL_IMPL[name](args)
            print(f"  Step{step} → {name}({args}) = {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

    print("⚠️ 达到最大步数")

react("台北人口是纽约人口的百分之几？保留整数。")
react("东京人口是多少？")