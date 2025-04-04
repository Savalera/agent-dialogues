"""Chat agent."""

from typing import Any, cast

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables.base import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from agentdialogues.exceptions import LLMInvocationError

# ⬇ Global runtime constants — set via init_agent)
llm: Runnable[list[BaseMessage], list[BaseMessage]]


# === Agent schema ===
ChatAgent = CompiledStateGraph


class ChatAgentConfig(BaseModel):
    """Chat agent configuration."""

    llm: Runnable[list[BaseMessage], list[BaseMessage]]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === Agent state ===
class ChatAgentState(BaseModel):
    """Chat state."""

    messages: list[AnyMessage]
    system_prompt: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === Graph nodes and edges ===
def chat_node(state: ChatAgentState) -> dict[str, Any]:
    """Dialogue chat node."""
    global llm

    system = SystemMessage(state.system_prompt)

    try:
        response = llm.invoke([system] + cast(list[BaseMessage], state.messages))
    except Exception as e:
        raise LLMInvocationError("LLM invocation failed in dialogue agent.") from e

    return {"messages": state.messages + [response]}


# === Graph builder ===
workflow = StateGraph(ChatAgentState)
workflow.add_node("agent", chat_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()


# === Agent creator ===
def create_chat_agent(config: dict[str, Any]) -> ChatAgent:
    """Create chat agent."""
    global llm

    agent_config = ChatAgentConfig(**config)
    llm = agent_config.llm

    return graph
