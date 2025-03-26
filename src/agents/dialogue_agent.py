"""Dialogue agent."""

from typing import Any, cast

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables.base import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from exceptions import LLMInvocationError


class DialogueAgentConfig(BaseModel):
    """Dialogue agent configuration."""

    llm: Runnable[list[BaseMessage], list[BaseMessage]]
    system_prompt: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class State(BaseModel):
    """Agent state."""

    messages: list[AnyMessage]
    llm: Runnable[list[BaseMessage], list[BaseMessage]]
    system_prompt: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


def chat_node(state: State) -> dict[str, Any]:
    """Dialogue chat node."""
    llm = state.llm
    system = SystemMessage(state.system_prompt)

    try:
        response = llm.invoke([system] + cast(list[BaseMessage], state.messages))
    except Exception as e:
        raise LLMInvocationError("LLM invocation failed in dialogue agent.") from e

    return {"messages": state.messages + [response]}


workflow = StateGraph(State)
workflow.add_node("agent", chat_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()
