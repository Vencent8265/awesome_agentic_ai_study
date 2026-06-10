import requests
import os
import json
from dotenv import load_dotenv

#加载.env里的变量
load_dotenv()

url = "https://api.github.com/user"

#不带token -- 应返回401
response_no_auth = requests.get(url)
print(f"无token状态码：{response_no_auth.status_code}")

#带 token -- 应返回200
token = os.getenv("GITHUB_TOKEN")
print(f"token是否加载：{token}")
headers = {"Authorization":f"Bearer {token}"}
response_with_auth = requests.get(url,headers=headers)
print("完整报错：",json.dumps(response_with_auth.json(),indent=2,ensure_ascii=False))  # 加这行看完整报错

print(f"带token状态码：{response_with_auth.status_code}")
print(f"登陆用户：{response_with_auth.json()['login']}")