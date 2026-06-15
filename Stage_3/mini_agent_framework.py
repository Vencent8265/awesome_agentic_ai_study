"""
目的：用最少的代码，看清楚Agent框架的骨架
包含：Tool基类/Tool Registry/Agent抽象基类/ReActAgent实现
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Callable, Any
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
# LAYER 1：工具系统
# 解决的问题：函数怎么变成有 name/description/schema 的可管理工具
# ══════════════════════════════════════════════════════════════

class Tool:
    """
    工具类：把一个普通的python函数包装成LLM可以理解和调用的工具

    核心思想：
    - fn：给python执行的
    - scheam：给LLM看的说明书
    """

    def __init__(self,name:str,description:str,fn:Callable,parameters:dict):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters #JSON Schema格式

    def run(self,args:dict) -> Any:
        """执行工具函数，args是LLM传来的参数dict"""
        return self.fn(**args)
    
    def to_schema(self) -> dict:
        """
        转成Anthropic tools格式的说明书
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
    
class ToolRegistry:
    """
    工具注册表。
    解决Agent怎么管理多个工具，怎么按名字找到并执行

    核心思想：一个dict，key = 工具名，value = Tool 对象
    """
    def __init__(self):
         self._tools:dict[str,Tool] = {}
    
    def register(self,tool:Tool):
        """注册一个工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名字取工具"""
        return self._tools.get(name)
 
    def execute(self, name: str, args: dict) -> str:
        """找到工具并执行，结果统一转成 string 还给 LLM"""
        tool = self.get(name)
        if not tool:
            return json.dumps({"error": f"tool '{name}' not found"})
        try:
            result = tool.run(args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "retry_hint": "check your arguments"})
 
    def schemas(self) -> list[dict]:
        """返回所有工具的 schema 列表，直接传给 LLM"""
        return [t.to_schema() for t in self._tools.values()]
    
# ══════════════════════════════════════════════════════════════
# LAYER 2：Agent 抽象基类
# 解决的问题：怎么让所有 Agent 类型遵守同一个接口
# ══════════════════════════════════════════════════════════════
 
class Agent(ABC):
    """
    Agent 抽象基类。
 
    核心思想：
    - 用 ABC + @abstractmethod 强制所有子类实现 run()
    - 公共的东西（LLM client、registry、历史）放在基类
    - 具体的 loop 逻辑放在子类
 
    这就是为什么 LangGraph / AutoGen 里的各种 Agent
    都有一个统一的 .invoke() / .run() 入口——基类定义的。
    """
 
    def __init__(self, name: str, registry: ToolRegistry, model: str, max_iter: int = 6):
        self.name = name
        self.registry = registry
        self.model = model
        self.max_iter = max_iter
        self.client = anthropic.Anthropic(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/anthropic",
        )
 
    @abstractmethod
    def run(self, query: str) -> str:
        """
        所有子类必须实现这个方法。
        @abstractmethod 保证你不能直接实例化 Agent()，
        必须用具体的子类（ReActAgent 等）。
        """
        pass
 
    def _call_llm(self, messages: list) -> object:
        """封装 LLM 调用，所有子类共用"""
        return self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=self.registry.schemas(),
            messages=messages,
        )
    
# ══════════════════════════════════════════════════════════════
# LAYER 3：ReActAgent 实现
# 解决的问题：继承基类，把 Stage 3 的 while loop 封装成类
# ══════════════════════════════════════════════════════════════
 
class ReActAgent(Agent):
    """
    ReAct Agent：继承 Agent 基类，实现 run() 方法。
 
    对比 Stage 3 的写法：
    - 之前：每次都重新写 messages = [...] 和 for step in range(...)
    - 现在：agent = ReActAgent(...); agent.run("问题")
    - loop 结构完全一样，只是封装进了类
    """
 
    def run(self, query: str) -> str:
        print(f"\n{'='*55}")
        print(f"[{self.name}] 问题：{query}")
        print(f"{'='*55}")
 
        messages = [{"role": "user", "content": query}]
 
        for step in range(self.max_iter):
            print(f"\n── Step {step + 1} ──────────────────────────")
 
            resp = self._call_llm(messages)
            print(f"stop_reason: {resp.stop_reason}")
 
            # assistant 回复接回 messages（loop 能运转的关键）
            messages.append({"role": "assistant", "content": resp.content})
 
            # 没有 tool_use → 任务完成，退出
            if resp.stop_reason == "end_turn":
                final = next((b.text for b in resp.content if hasattr(b, "text")), "")
                print(f"\n✅ 最终回答：{final}")
                return final
 
            # 有 tool_use → 执行工具，把结果还给 LLM
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                print(f"调用工具：{block.name}，参数：{block.input}")
                obs = self.registry.execute(block.name, block.input)
                print(f"工具返回：{obs}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": obs,
                })
 
            messages.append({"role": "user", "content": tool_results})
 
        return "已达到最大步骤数，任务未完成。"
    
# ══════════════════════════════════════════════════════════════
# 使用示例：组装框架，跑一个 agent
# ══════════════════════════════════════════════════════════════
 
def make_calculator_tool() -> Tool:
    """把计算函数包装成 Tool 对象"""
    def calculate(expression: str) -> dict:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e), "retry_hint": "check the expression syntax"}
 
    return Tool(
        name="calculator",
        description="Evaluate a math expression. Use for any arithmetic calculation.",
        fn=calculate,
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '(19 * 42) - 8'"},
            },
            "required": ["expression"],
        },
    )
 
 
def make_weather_tool() -> Tool:
    """把天气函数包装成 Tool 对象"""
    def get_weather(city: str) -> dict:
        fake = {
            "台北": {"forecast": "rain", "temperature_c": 24},
            "Tokyo": {"forecast": "sunny", "temperature_c": 28},
        }
        return fake.get(city, {"forecast": "cloudy", "temperature_c": 22})
 
    return Tool(
        name="get_weather",
        description="Get current weather for a city. Use when asked about weather or temperature.",
        fn=get_weather,
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    )
 
 
if __name__ == "__main__":
    # 1. 注册工具
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    registry.register(make_weather_tool())
 
    # 2. 创建 Agent（传入 registry，不需要关心工具细节）
    agent = ReActAgent(
        name="MyAgent",
        registry=registry,
        model="deepseek-v4-pro",
    )
 
    # 3. 跑起来
    agent.run("台北现在几度？再帮我算一下这个温度乘以 9/5 加 32 等于几华氏度？")