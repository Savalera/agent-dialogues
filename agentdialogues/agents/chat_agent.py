"""Chat agent."""

from typing import Any, Optional, cast

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables.base import Runnable
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from agentdialogues.core.base import ChatProviders
from agentdialogues.exceptions import LLMInvocationError


# === Agent state schema ===
class AgentConfig(BaseModel):
    """Agent configuration schema."""

    model_name: str
    provider: ChatProviders = Field(default=ChatProviders.OLLAMA)
    seed: int


class Runtime(BaseModel):
    """Agent runtime configuration."""

    llm: Runnable[list[BaseMessage], list[BaseMessage]]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentState(BaseModel):
    """Chat state."""

    messages: list[AnyMessage] = []
    system_prompt: str = ""
    raw_config: dict[str, Any] = Field(default_factory=dict)
    config: Optional[AgentConfig] = None
    runtime: Optional[Runtime] = None


# === Graph nodes and edges ===
def setup_node(state: AgentState) -> dict[str, Any]:
    """Set up agent state."""
    config = AgentConfig(**state.raw_config)

    llm = cast(
        Runnable[list[BaseMessage], list[BaseMessage]],
        ChatOllama(model=config.model_name, seed=config.seed),
    )

    return {"config": config, "runtime": {"llm": llm}}


def chat_node(state: AgentState) -> dict[str, Any]:
    """Dialogue chat node."""
    assert state.runtime

    system = SystemMessage(state.system_prompt)

    try:
        response = state.runtime.llm.invoke(
            [system] + cast(list[BaseMessage], state.messages)
        )
    except Exception as e:
        raise LLMInvocationError("LLM invocation failed in dialogue agent.") from e

    return {"messages": state.messages + [response]}


# === Graph builder ===
workflow = StateGraph(AgentState)
workflow.add_node("setup", setup_node)
workflow.add_node("chat", chat_node)

workflow.add_edge(START, "setup")
workflow.add_edge("setup", "chat")
workflow.add_edge("chat", END)

graph = workflow.compile()
