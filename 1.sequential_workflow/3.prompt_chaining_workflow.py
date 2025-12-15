from langgraph.graph import StateGraph, START,END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class llm_chain(TypedDict):
    title: str
    outline: str
    blog: str

def create_outline(state: llm_chain) -> llm_chain:
    title = state['title']

    prompt = f'Generate a detailed outline for a blog on the topic - {title}'

    outline = model.invoke(prompt).content

    state['outline'] = outline

    return state

def create_blog(state: llm_chain) -> llm_chain:
    title = state['title']
    outline = state['outline']

    prompt = f'Write a detailed blog on the title - {title} using the follwing outline \n {outline}'

    blog = model.invoke(prompt).content

    state['blog'] = blog

    return state

graph = StateGraph(llm_chain)

graph.add_node('outline',create_outline)
graph.add_node('blog', create_blog)

graph.add_edge(START,'outline')
graph.add_edge('outline','blog')
graph.add_edge('blog',END)

workflow = graph.compile()

intial_state = {'title': 'Rise of AI in India'}

final_result = workflow.invoke(intial_state)

print(final_result['outline'])