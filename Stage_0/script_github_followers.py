#用Python发送一个GET请求到GitHub API，解析返回的JSON，打印出Linus Torvalds的followers数量

import requests

#1.发送GET请求到GitHub API
url = "https://api.github.com/users/torvalds"
response = requests.get(url)

#2.解析JSON响应,将返回的json值存储为字典data
data = response.json()
print(data)

#3.打印followers数量
print(f"Linus Torvalds 的 followers 数量：{data['followers']}")
print(f"Linus Torvalds 的 公司名是：{data['company']}")