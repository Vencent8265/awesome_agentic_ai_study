import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 正常客户端
# ============================================================
client = anthropic.Anthropic(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/anthropic"
)

# ============================================================
# 场景1：错误的 API Key → 401
# ============================================================
print("=== 场景1：错误的 API Key ===")
try:
    bad_client = anthropic.Anthropic(
        api_key="sk-wrong-key-123",
        base_url="https://api.deepseek.com/anthropic"
    )
    bad_client.messages.create(
        model="deepseek-v4-flash",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}]
    )
except anthropic.AuthenticationError as e:
    print(f"✅ 捕获到认证错误：{e.status_code}")
except Exception as e:
    print(f"✅ 捕获到错误：{type(e).__name__}: {e}")

# ============================================================
# 场景2：max_tokens 设成1，故意截断输出
# ============================================================
print("\n=== 场景2：max_tokens=1 截断输出 ===")
msg = client.messages.create(
    model="deepseek-v4-flash",
    max_tokens=1,
    messages=[{"role": "user", "content": "用100字介绍自己"}]
)
print(f"stop_reason: {msg.stop_reason}")  # 应该是 max_tokens
# 替换这两行
text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "（无输出）")
print(f"输出内容: {text}")
if msg.stop_reason == "max_tokens":
    print("✅ 检测到输出被截断，应增大 max_tokens")

# ============================================================
# 场景3：带 exponential backoff 的 retry wrapper
# ============================================================
print("\n=== 场景3：Retry Wrapper ===")

def call_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="deepseek-v4-flash",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            return next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)

        except anthropic.RateLimitError:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"  触发限速，{wait}s 后重试（第{attempt+1}次）")
            time.sleep(wait)

        except anthropic.APIConnectionError:
            wait = 2 ** attempt
            print(f"  网络错误，{wait}s 后重试（第{attempt+1}次）")
            time.sleep(wait)

        except Exception as e:
            print(f"  未知错误：{type(e).__name__}: {e}")
            break

    return None  # 全部重试失败

result = call_with_retry("用一句话介绍DeepSeek")
if result:
    print(f"✅ 调用成功：{result[:80]}")
else:
    print("❌ 全部重试失败")