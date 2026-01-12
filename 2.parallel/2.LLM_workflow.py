from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import TypedDict,Annotated
import operator
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class Evalutionschema(BaseModel):

    feedback : str = Field(description="Detailed feedbackfor the essay")
    score : int = Field(description="score out of 10",ge= 0, le=10)

structure_model = model.with_structured_output(Evalutionschema)

essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""

prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essay}'
final1 = structure_model.invoke(prompt).feedback

# print(final1)

class UPSCState(TypedDict):

    essay : str
    langage_feedback : str
    analysis_feedback : str
    clarity_feedback : str
    overall_feedbacK : str
    individual_scores : Annotated[list[int],operator.add]
    avg_score : float

def evaluate_langauge(state: UPSCState):
    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    result = structure_model.invoke(prompt)

    return {
        'lanagauge_feedback': result.feedback,
        'individual_scores': [result.score]
    }

def evaluate_analysis(state: UPSCState):
    prompt = f'Evaluate the analytical quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    result = structure_model(prompt)

    return {
        'analysis_feedback': result.feedbacj,
        'individual_scores': [result.score]
    }

def evaluate_thought(state: UPSCState):

    prompt = f'Evaluate the clarity of thought in the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    result = structure_model(prompt)

    return {
        'clarity_feedback': result.feedback,
        'individual_scores': [result.score]
    }

def final_evaluation(state: UPSCState):
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt)

    avg_score = sum(state['individual_scores'])/ len(state['individuial_scores'])
    return {
        'overall_feedback': overall_feedback,
         'avg_score': avg_score
    }


graph = StateGraph(UPSCState)
graph.add_node('evaluate_lanagauge', evaluate_langauge)
graph.add_node('evaluate_analysis',evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

graph.add_edge(START,'evaluate_lanagauge')
graph.add_edge(START,'evaluate_analysis')
graph.add_edge(START,'evaluate_thought')

graph.add_edge('evaluate_lanagauge','final_evaluation')
graph.add_edge('evaluate_analysis','final_evaluation')
graph.add_edge('evaluate_thought','final_evaluation')

graph.add_edge('fianl_evaluation', END)

workflow = graph.compile()

initial_state : UPSCState = {
    'essay': essay
}

workflow_result = workflow.invoke(initial_state)

