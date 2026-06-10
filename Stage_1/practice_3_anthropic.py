import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 价格表（USD / 1M tokens，2026-06 snapshot）
# 来源：api-docs.deepseek.com / openrouter.ai
# ============================================================
PRICING = {
    "deepseek-v4-pro": {
        "input":  0.435,   # $0.435/1M input tokens（75%折扣价，已成正式价）
        "output": 0.870,   # $0.870/1M output tokens
        "note":   "DeepSeek flagship，强推理，低价"
    },
    "deepseek-v4-flash": {
        "input":  0.14,    # $0.14/1M input tokens
        "output": 0.28,    # $0.28/1M output tokens
        "note":   "DeepSeek 快速版，速度快，价格极低"
    },
}

PROMPT = "用3句话解释什么是大语言模型。"

results = []

# ============================================================
# DeepSeek 模型（via Anthropic SDK）
# ============================================================
deepseek_client = anthropic.Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/anthropic"
)

for model in ["deepseek-v4-pro", "deepseek-v4-flash"]:
    # max_tok = 1000 if model == "deepseek-v4-pro" else 200
    t0 = time.time()
    msg = deepseek_client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT}]
    )

    print("=== DEBUG Pro content ===")
    for i, block in enumerate(msg.content):
        print(f"block[{i}]: type={block.type}")
        if hasattr(block, 'text'):
            print(f"  text前50字: {block.text[:50]}")
        if hasattr(block, 'thinking'):
            print(f"  thinking前50字: {block.thinking[:50]}")
    print("=== END DEBUG ===")

    latency = time.time() - t0

    in_tok  = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens
    price   = PRICING[model]
    cost    = (in_tok * price["input"] + out_tok * price["output"]) / 1_000_000

    # 跳过 thinking block，只取 text block
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")


    results.append({
        "model":   model,
        "latency": latency,
        "in_tok":  in_tok,
        "out_tok": out_tok,
        "cost":    cost,
        "text":    text,
        "note":    price["note"]
    })

# ============================================================
# 打印对比结果
# ============================================================
print(f"\n{'='*60}")
print(f"Prompt: {PROMPT}")
print(f"{'='*60}\n")

for r in results:
    print(f"【{r['model']}】")
    print(f"  说明：{r['note']}")
    print(f"  耗时：{r['latency']:.2f}s")
    print(f"  Tokens：input={r['in_tok']}  output={r['out_tok']}")
    print(f"  本次费用：${r['cost']:.6f}")
    print(f"  1000次费用：${r['cost']*1000:.4f}")
    print(f"  回答：{r['text'][:500]}...")
    print()