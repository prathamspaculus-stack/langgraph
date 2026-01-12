from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: chatstate):

    messages = state["messages"]
    response = model.invoke(messages)

    return {
        "messages": [response]
    }

graph = StateGraph(chatstate)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

conn = sqlite3.connect(database = 'ms.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
# Streaming example

# for message_chunk, metadata in chatbot.stream(
#     {'messages': [HumanMessage(content="what is a receipe of pasta")] },
#     config=config,
#     stream_mode='messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=' ', flush = True)



# while True:

#     user_message = input("you:")

#     if user_message.lower() in ['exit','quite', 'bye']:
#         break

#     config1 = {"configurable": {"thread_id": "1"}}

#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config1)

#     print("bot:", response['messages'][-1].content)

