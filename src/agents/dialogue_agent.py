"""Dialogue agent."""

from typing import List

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from exceptions import LLMInvocationError


class State(BaseModel):
    """Agent state."""

    messages: List[AnyMessage]
    llm: Runnable
    system_prompt: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


def chat_node(state: State):
    """Dialogue chat node."""
    llm = state.llm
    system = SystemMessage(state.system_prompt)

    try:
        response = llm.invoke([system] + state.messages)
    except Exception as e:
        raise LLMInvocationError("LLM invocation failed in dialogue agent.") from e

    return {"messages": state.messages + [response]}


workflow = StateGraph(State)
workflow.add_node("agent", chat_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()
