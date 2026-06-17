from typing import TypedDict, List
from langchain_core.messages import HumanMessage
# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# response = llm.invoke("你好，介绍一下你自己")
# print(response)

class AgentState(TypedDict):
    message: List[HumanMessage]

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"\nAI:{response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter:")
while user_input != "exit":
    agent.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("Enter:")