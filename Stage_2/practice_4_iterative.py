# Stage 2 - 练习4：Iterative Refinement（迭代优化prompt）
# 运行：python3 practice_4.py

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gemma4:e4b"

# 5个iteration，每一轮prompt都比前一轮更具体
PROMPTS = {
    "v1 模糊": "介绍一下ReAct。",
    "v2 加目标读者": "介绍一下ReAct，给写过Python的软件工程师看。",
    "v3 加格式": "介绍一下ReAct，给写过Python的软件工程师看。100字以内，用一个段落。",
    "v4 加example要求": "介绍一下ReAct，给写过Python的软件工程师看。100字以内，用一个段落，结尾举一个具体例子（比如查天气）。",
    "v5 加禁忌": "介绍一下ReAct，给写过Python的软件工程师看。100字以内，用一个段落，结尾举一个具体例子（比如查天气）。不要用'赋能'、'驱动'、'智能'这类空泛词汇。",
}

outputs = {}
for label, prompt in PROMPTS.items():
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    msg = r.choices[0].message
    text = msg.content or getattr(msg, "reasoning", "") or ""
    outputs[label] = text
    print(f"\n--- [{label}] ({len(text)}字) ---")
    print(text)

# 验证
print("\n" + "=" * 50)
banned = ("赋能", "驱动", "智能")
v5_banned = [w for w in banned if w in outputs["v5 加禁忌"]]

if v5_banned:
    print(f"⚠️  v5仍含禁忌词：{v5_banned}（小模型遵循度有限，记下来）")
else:
    print("✅ v5无禁忌词")

print(f"\n✅ 练习4完成")
print(f"   v1长度：{len(outputs['v1 模糊'])}字  →  v5长度：{len(outputs['v5 加禁忌'])}字")
print()
print("💡 观察重点：")
print("   - 每加一个约束，输出就收敛一次")
print("   - v1 vs v5 的差距就是'prompt质量'的差距")
print("   - 这个迭代过程就是真实工作中写prompt的方式")