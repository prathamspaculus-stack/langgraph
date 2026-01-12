from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class Batsmanstate(TypedDict):

    runs: int
    balls: int
    four: int
    six: int

    sr : float
    bpb : float
    boundary_percentage : int
    summary : str

def calculate_sr(state: Batsmanstate):

    runs = state['runs']
    balls = state['balls']

    sr = (runs/balls)*100

    return {'sr': sr}

def calculate_bpb(state: Batsmanstate):

    bpb = state['balls']/(state['four'] + state['six'])

    return {'bpb': bpb}
def calculate_boundary_percent(state: Batsmanstate):

    boundary_percent = (((state['four'] * 4) + (state['six'] * 6))/state['runs'])*100

    return {'boundary_percentage': boundary_percent}
def summary(state: Batsmanstate):

    summary = f"""
Strike Rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percent - {state['boundary_percentage']}
"""
    
    return {'summary': summary}

graph = StateGraph(Batsmanstate)

graph.add_node('sr', calculate_sr)
graph.add_node('bpb', calculate_bpb)
graph.add_node('bp', calculate_boundary_percent)
graph.add_node('summary', summary)

graph.add_edge(START,'sr')
graph.add_edge(START,'bpb')
graph.add_edge(START,'bp')

graph.add_edge('sr','summary')
graph.add_edge('bpb','summary')
graph.add_edge('bp','summary')

graph.add_edge('summary',END)

workflow = graph.compile()

values = {
    'runs': 100,
    'balls': 50,
    'four': 5,
    'six': 5
}

final_values = workflow.invoke(values)

print(final_values['summary'])
