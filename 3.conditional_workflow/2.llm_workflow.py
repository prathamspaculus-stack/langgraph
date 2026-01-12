from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model='llama-3.1-8b-instant')


class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative']

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "frustrated", "disappointed", "calm"]
    urgency: Literal["low", "medium", "high"]

sentiment_model = model.with_structured_output(SentimentSchema)
diagnosis_model = model.with_structured_output(DiagnosisSchema)


class ReviewState(TypedDict, total=False):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


def analyze_sentiment(state: ReviewState):
    prompt = f"Find the sentiment of this review:\n{state['review']}"
    result = sentiment_model.invoke(prompt)
    return {"sentiment": result.sentiment}

def check(state: ReviewState) -> Literal['positive_response', 'run_diagnosis']:
    return 'positive_response' if state['sentiment'] == 'positive' else 'run_diagnosis'

def positive_response(state: ReviewState):
    prompt = f"""
Write a warm thank-you message for this review:

"{state['review']}"

Also ask the user to leave feedback on our website.
"""
    response = model.invoke(prompt)
    return {"response": response.content}

def run_diagnosis(state: ReviewState):
    prompt = f"""
Diagnose this negative review:

{state['review']}

Return issue_type, tone, and urgency.
"""
    result = diagnosis_model.invoke(prompt)
    return {"diagnosis": result.model_dump()}

def negative_response(state: ReviewState):
    d = state['diagnosis']
    prompt = f"""
You are a support assistant.

Issue: {d['issue_type']}
Tone: {d['tone']}
Urgency: {d['urgency']}

Write an empathetic and helpful resolution.
"""
    response = model.invoke(prompt)
    return {"response": response.content}


graph = StateGraph(ReviewState)

graph.add_node('analyze_sentiment', analyze_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

graph.add_edge(START, 'analyze_sentiment')
graph.add_conditional_edges('analyze_sentiment', check)
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)

workflow = graph.compile()


initial_state = {
    "review": "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen."
}

print(workflow.invoke(initial_state))
