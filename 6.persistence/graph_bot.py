from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage

load_dotenv()

model= ChatGroq(model="llama-3.1-8b-instant")

class reviewstate(TypedDict):
    topic: str
    joke: str
    explainantion: str


def joke(state: reviewstate):
    topic = state['topic']
    prompt = f"Tell me a joke about {topic}"
    response = model.invoke(prompt).content

    return {'joke': response} 

def explain(state: reviewstate):

    joke = state['joke']

    prompt = f"Explain the following joke in spimpler terms: {joke}"

    response = model.invoke(prompt).content

    return {"explainantion": response}

graph = StateGraph(reviewstate)

graph.add_node('joke', joke)
graph.add_node('explain', explain)

graph.add_edge(START, 'joke')
graph.add_edge('joke', 'explain')
graph.add_edge('explain', END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

# while True:

#     user_topic = input("Enter a topic for a joke: ")

#     if user_topic.lower() in ['exit', 'quite', 'bye']:
#         break

#     config = {"configurable": {"thread_id": 1}}
#     response = workflow.invoke({'topic': [HumanMessage(content=user_topic)] }, config=config)

#     print("joke", response)

# print(workflow.get_state(config))

# print(list(workflow.get_state_history(config)))

# workflow.get_state({"configurable": {"thread_id": "1", "checkpoint_id": "1f0ec614-60cf-612a-8002-9ea82373e027"}})