from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "need_improvement"]
    feedback: str

structured_model = model.with_structured_output(TweetEvaluation)

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "need_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

def generator(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Max 280 characters
- No Q&A format
- Simple, day to day English
- Meme logic, sarcasm, or irony
""")
    ]

    tweet = model.invoke(messages).content

    return {
        "tweet": tweet,
        "iteration": state["iteration"],
        "topic": state["topic"],
        "max_iteration": state["max_iteration"]
    }

def evaluator(state: TweetState):
    messages = [
        SystemMessage(content="You are a ruthless Twitter critic."),
        HumanMessage(content=f"""
Evaluate this tweet:

"{state['tweet']}"

Respond ONLY in structured format:
- evaluation: approved or need_improvement
- feedback: one short paragraph
""")
    ]

    response = structured_model.invoke(messages)

    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback
    }

def optimizer(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality."),
        HumanMessage(content=f"""
Improve this tweet based on feedback:

Feedback:
{state['feedback']}

Original Tweet:
{state['tweet']}
""")
    ]

    improved = model.invoke(messages).content

    return {
        "tweet": improved,
        "iteration": state["iteration"] + 1
    }

def route_evaluate(state: TweetState):
    if state["evaluation"] == "approved" or state["iteration"] >= state["max_iteration"]:
        return "approved"
    return "need_improvement"

graph = StateGraph(TweetState)

graph.add_node("generator", generator)
graph.add_node("evaluator", evaluator)
graph.add_node("optimizer", optimizer)

graph.add_edge(START, "generator")
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges(
    "evaluator",
    route_evaluate,
    {
        "approved": END,
        "need_improvement": "optimizer"
    }
)
graph.add_edge("optimizer", "evaluator")  

workflow = graph.compile()

initial_state = {
    "topic": "gkujhk",
    "iteration": 0,
    "max_iteration": 5
}

result = workflow.invoke(initial_state)
print(result)
