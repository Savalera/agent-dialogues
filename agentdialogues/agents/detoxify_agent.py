"""Detoxify agent."""

from typing import Any, Literal, Optional

from detoxify import Detoxify
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentdialogues.exceptions import LLMInvocationError

# === Agent state schema ===
DetoxifyModelType = Literal[
    "original", "unbiased", "multilingual", "original-small", "unbiased-small"
]


class AgentConfig(BaseModel):
    """Agent configuration schema."""

    model: DetoxifyModelType = "original"
    device: str


class ToxicityScore(BaseModel):
    """Toxicity score."""

    toxicity: float
    severe_toxicity: float
    obscene: float
    threat: float
    insult: float
    identity_attack: float
    sexual_explicit: Optional[float] = None


class Evaluation(BaseModel):
    """Toxicity evaluation."""

    evaluator: Literal["Detoxify"] = "Detoxify"
    model: DetoxifyModelType
    score: ToxicityScore


class AgentState(BaseModel):
    """Agent state."""

    message: str
    evaluation: Optional[Evaluation] = None
    raw_config: dict[str, Any] = Field(default_factory=dict)
    config: Optional[AgentConfig] = None


# === Graph nodes and edges ===
def setup_node(state: AgentState) -> dict[str, Any]:
    """Set up agent state."""
    config = AgentConfig(**state.raw_config)

    return {"config": config}


def detoxify_node(state: AgentState) -> dict[str, Any]:
    """Detoxify evaluation node."""
    try:
        assert state.config

        results = Detoxify(state.config.model, device=state.config.device).predict(
            state.message
        )
    except Exception as e:
        raise LLMInvocationError("LLM invocation failed in dialogue agent.") from e

    return {
        "evaluation": Evaluation(
            model=state.config.model, score=ToxicityScore(**results)
        )
    }


# === Graph builder ===
workflow = StateGraph(AgentState)
workflow.add_node("detoxify", detoxify_node)
workflow.add_node("setup", setup_node)

workflow.add_edge(START, "setup")
workflow.add_edge("setup", "detoxify")
workflow.add_edge("detoxify", END)

graph = workflow.compile()
