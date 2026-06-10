# Stage 2 - 练习2：Few-Shot
# 需要：pip install openai
# 前置：ollama serve 已在后台运行
# 运行：python practice_2.py

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL = "qwen2.5:3b"

# 中文情绪分类（正面 / 负面 / 中立）
TEST_SET = [
    ("这部电影超赞、看完想再看一次！", "正面"),
    ("剧情无聊、演员演技尴尬。", "负面"),
    ("这是一部2019年的电影。", "中立"),
    ("我不确定喜不喜欢、可能再想想。", "中立"),
    ("第一集很不错但第二集就崩了。", "负面"),
    ("看完心情很好、推荐！", "正面"),
]

FEW_SHOT_EXAMPLES = """范例：
input: 这家餐厅的牛排好吃到让我哭出来。
output: 正面

input: 服务生态度很差、我再也不会来了。
output: 负面

input: 这家店位于新北市三重区。
output: 中立

"""


def classify(text: str, *, use_few_shot: bool) -> str:
    prefix = FEW_SHOT_EXAMPLES if use_few_shot else ""
    prompt = f"{prefix}input: {text}\noutput:"
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip().splitlines()[0]


def evaluate(use_few_shot: bool) -> tuple:
    correct = 0
    for text, label in TEST_SET:
        pred = classify(text, use_few_shot=use_few_shot)
        ok = label in pred
        print(f"  {'✓' if ok else '✗'} 预期[{label}] | 实际[{pred}] | {text[:20]}...")
        if ok:
            correct += 1
    return correct, len(TEST_SET)


print("=" * 50)
print("=== 0-shot（没有范例）===")
print("=" * 50)
c0, n = evaluate(use_few_shot=False)
print(f"结果：{c0}/{n} = {c0/n:.0%}\n")

print("=" * 50)
print("=== 3-shot（给了3个范例）===")
print("=" * 50)
c3, _ = evaluate(use_few_shot=True)
print(f"结果：{c3}/{n} = {c3/n:.0%}\n")

# === 自我验证 ===
print("=" * 50)
assert c3 >= c0, f"预期3-shot不比0-shot差，实际 {c3} < {c0}"
improvement = c3 - c0
print(f"✅ 练习2通过")
print(f"   0-shot：{c0}/{n}  →  3-shot：{c3}/{n}  （提升了{improvement}题）")
print()
print("💡 观察重点：")
print("   - '中立'这类模糊情绪，0-shot最容易判错")
print("   - 给了范例之后，模型知道'中立'是一个合法答案")
print("   - few-shot 本质是：用例子告诉模型'你期望什么格式和逻辑'")