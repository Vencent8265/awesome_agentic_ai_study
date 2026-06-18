from typing import TypedDict, List,Union
from langchain_core.messages import HumanMessage, AIMessage
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

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    state["message"].append(AIMessage(content=response.content))
    print(f"\nAI:{response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter:")
conversation_history = []
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"message": conversation_history})
    conversation_history = result["message"]
    user_input = input("Enter:")

with open("3to4_LangGraph/Agent/logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")
print("Conversation saved to logging.txt")