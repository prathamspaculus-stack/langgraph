from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
import requests
import random


load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

search_tool = DuckDuckGoSearchRun(region ='us-en')

@tool
def calculator(first_num : float, second_num: float, operation: str) -> dict:
    """perform a basic arithemetic operation on two numbers. Supported operations are: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=IHVZQ9X6NJFWYIX9"
    r = requests.get(url)
    return r.json()

tools = [get_stock_price, calculator, search_tool]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node (state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")  

chatbot = graph.compile()

out = chatbot.invoke({"messages": [HumanMessage(content="What is 2*3?")]})
print(out["messages"][-1].content)