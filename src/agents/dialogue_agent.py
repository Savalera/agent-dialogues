"""Dialogue agent."""

from typing import Any

from langchain_core.messages import (
    SystemMessage,
)
from langgraph.graph import END, START, MessagesState, StateGraph


class State(MessagesState):
    """Agent state."""

    llm: Any
    role: str
    system_prompt: str


def chat_node(state: State):
    """Dialogue chat node."""
    llm = state["llm"]
    system = SystemMessage(state["system_prompt"])

    response = llm.invoke([system] + state["messages"])

    return {"messages": response}


workflow = StateGraph(State)
workflow.add_node("agent", chat_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()
