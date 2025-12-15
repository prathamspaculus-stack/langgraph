from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from IPython.display import Image



class BMIstate(TypedDict):
    weight_kg : float
    height_m  : float
    BMI: float
    category: str

def calculate_bmi(state : BMIstate) -> BMIstate:
    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight/(height**2)

    state['BMI'] = round(bmi,2) 

    return state


def Label_bmi(state : BMIstate) -> BMIstate:
    bmi = state['BMI']

    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"

    return state

graph = StateGraph(BMIstate)

graph.add_node('calculate_bmi',calculate_bmi)
graph.add_node('Label_bmi',Label_bmi)

graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi','Label_bmi')
graph.add_edge('Label_bmi',END)

workflow = graph.compile()
workflow

initial_state = {'weight_kg':80, 'height_m': 1.73}

final_state = workflow.invoke(initial_state)

print(Image(workflow.get_graph().draw_mermaid_png()))

print(final_state)