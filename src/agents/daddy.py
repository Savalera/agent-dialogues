"""Daddy agent."""

from langchain_core.messages import (
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph

llm = ChatOllama(model="mistral-nemo")


def chat_node(state: MessagesState):
    """Daddy chat node."""
    system = SystemMessage(
        content="""
        You are a wise and experienced father 
        answering the questions of your baby with patience.

        Your answers are loving, detailed, but not too long, just long enough for a child to grasp.
        """
    )

    response = llm.invoke([system] + state["messages"])
    response.type = "daddy"

    return {"messages": response}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", chat_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()
