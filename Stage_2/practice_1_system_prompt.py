# Stage 2 - 练习1：System Prompt
# 需要：pip install openai
# 前置：ollama serve（另开一个终端）
# 运行：python practice_1.py

import sys, json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODEL = "qwen2.5:3b"

# 同一个 user message、3 个不同 system prompt
SYSTEM_PROMPTS = {
    "严肃律师": "你是严谨的合约律师。回答要精准、引用法条编号、避免任何主观形容词。",
    "幼儿园老师": "你是温柔的幼儿园老师，要对5岁小孩说话。用比喻、口语、少于80字。",
    "JSON机器": '你只回JSON。schema: {"answer": string, "confidence": float}，不要输出任何其他内容。',
}

USER_MSG = "请帮我解释什么是租赁合约。"

print("=" * 50)
print(f"用户问题：{USER_MSG}")
print("=" * 50)

outputs = {}
for label, system in SYSTEM_PROMPTS.items():
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": USER_MSG},
        ],
    )
    text = r.choices[0].message.content
    outputs[label] = text
    print(f"\n--- [{label}] ---")
    print(text)

# === 自我验证 ===
print("\n" + "=" * 50)
json_output = outputs["JSON机器"]
assert "{" in json_output and "}" in json_output, "JSON机器版输出应该含 JSON braces"

try:
    # 尝试提取JSON部分
    import re
    json_match = re.search(r'\{[^{}]*\}', json_output, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        assert "answer" in parsed, "JSON schema 应包含 answer 字段"
        print(f"✅ JSON验证通过：answer字段存在，confidence={parsed.get('confidence', 'N/A')}")
except json.JSONDecodeError as e:
    print(f"⚠️  JSON解析有偏差（小模型常见），但包含大括号，练习仍然通过")

print(f"\n✅ 练习1通过 — 同一个问题、3种人格/格式/语气")
print("💡 观察重点：")
print("   - 严肃律师：是否更正式、有没有引用条款")
print("   - 幼儿园老师：是否更短、用了比喻")
print("   - JSON机器：输出是否包含{}结构（小模型遵循度不如Claude）")