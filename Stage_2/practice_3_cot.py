# Stage 2 - 练习3：Chain of Thought (CoT)
# 运行：python3 practice_3.py

import sys, re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:3b"

QUESTION = "小明有3颗苹果。他给了小华1颗、又从妈妈那边拿到5颗、然后吃了2颗。请问现在剩几颗？"
ANSWER = 5  # 3 - 1 + 5 - 2 = 5

COT_EXAMPLE = """范例：
Q: 一只鸡有2只脚。3只鸡跟1个人共有几只脚？
A: 让我一步一步算。3只鸡 × 2只脚 = 6只脚。1个人有2只脚。总共 6 + 2 = 8只脚。答案是8。

"""

def ask(prompt: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content

def extract_number(text: str):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None

# A. 纯prompt
out_a = ask(QUESTION)
ans_a = extract_number(out_a)

# B. + Let's think step by step
out_b = ask(QUESTION + "\n请一步一步思考。")
ans_b = extract_number(out_b)

# C. + CoT example
out_c = ask(COT_EXAMPLE + "Q: " + QUESTION + "\nA:")
ans_c = extract_number(out_c)

print("=" * 50)
for label, out, ans in [
    ("A 纯prompt", out_a, ans_a),
    ("B +step-by-step", out_b, ans_b),
    ("C +CoT example", out_c, ans_c),
]:
    print(f"\n--- [{label}] 答案={ans} {'✓' if ans == ANSWER else '✗'} ---")
    print(out[:300])

print("\n" + "=" * 50)
correct = sum(1 for a in (ans_a, ans_b, ans_c) if a == ANSWER)
print(f"✅ 练习3完成 — {correct}/3 答对")
print()
print("💡 观察重点：")
print("   - A 纯prompt：模型直接给答案，对不对全凭运气")
print("   - B +step-by-step：模型被迫写出推理过程，准确率提升")
print("   - C +CoT example：给了推理模板，最稳定")
print("   - 关键：CoT 不是让模型'更聪明'，而是让它'把草稿写出来'")