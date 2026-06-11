# Stage 3 - 练习1：Tool Use 基础
# 模型：deepseek-v4-pro via OpenAI SDK
# 运行：python3 practice_1.py

import sys, json
from dotenv import load_dotenv
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gemma4:e4b"

# ===== 1. 定义工具 =====
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行简单的数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：2 + 3 * 4"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ===== 2. 真正执行工具的函数（你来跑，不是模型）=====
def get_weather(city: str) -> str:
    # 模拟天气API，真实项目会调用OpenWeather等
    fake_data = {
        "北京": "晴，25°C，湿度40%",
        "上海": "多云，22°C，湿度65%",
        "广州": "小雨，28°C，湿度80%",
    }
    return fake_data.get(city, f"{city}：暂无数据")

def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

def execute_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return get_weather(**args)
    elif name == "calculate":
        return calculate(**args)
    return "未知工具"

# ===== 3. 对话主循环 =====
def chat_with_tools(user_message: str):
    print(f"\n用户：{user_message}")
    messages = [{"role": "user", "content": user_message}]

    # 第一次请求：模型决定要不要用工具
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        tools=tools,
        messages=messages,
    )

    msg = response.choices[0].message

    # 取文本（v4-pro是思考模型，content可能在reasoning_content里）
    text = msg.content or getattr(msg, "reasoning_content", "") or ""

    # 判断模型有没有要调工具
    if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"\n模型决定调用工具：{name}，参数：{args}")

            # 你来执行
            result = execute_tool(name, args)
            print(f"工具返回结果：{result}")

            # 把结果还给模型
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # 第二次请求：模型拿到结果，给出最终回答
        final = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
        )
        final_msg = final.choices[0].message
        final_text = final_msg.content or getattr(final_msg, "reasoning_content", "") or ""
        print(f"\n模型最终回答：{final_text}")
    else:
        # 模型直接回答，没有调工具
        print(f"\n模型直接回答（未调工具）：{text}")

# ===== 4. 测试三种场景 =====
print("=" * 50)
chat_with_tools("北京今天天气怎么样？")

print("=" * 50)
chat_with_tools("帮我算一下 (18 + 6) * 3 等于多少")

print("=" * 50)
chat_with_tools("你叫什么名字？")  # 这个不需要工具