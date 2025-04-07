"""Detoxify agent."""

from typing import Any, Literal, Optional

from detoxify import Detoxify
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from agentdialogues.exceptions import LLMInvocationError


# === Agent state ===
class ToxicityScore(BaseModel):
    """Toxicity score."""

    toxicity: float
    severe_toxicity: float
    obscene: float
    threat: float
    insult: float
    identity_attack: float


class Evaluation(BaseModel):
    """Toxicity evaluation."""

    evaluator: Literal["Detoxify"] = "Detoxify"
    model: Literal["original", "unbiased", "multilingual"] = "original"
    score: ToxicityScore


class DetoxifyAgentState(BaseModel):
    """Agent state."""

    message: str
    model: Literal["original", "unbiased", "multilingual"] = "original"
    device: str = "cpu"
    evaluation: Optional[Evaluation] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === Graph nodes and edges ===
def detoxify_node(state: DetoxifyAgentState) -> dict[str, Any]:
    """Detoxify evaluation node."""
    try:
        results = Detoxify(state.model, device=state.device).predict(state.message)
    except Exception as e:
        raise LLMInvocationError("LLM invocation failed in dialogue agent.") from e

    return {"evaluation": Evaluation(model=state.model, score=ToxicityScore(**results))}


# === Graph builder ===
workflow = StateGraph(DetoxifyAgentState)
workflow.add_node("agent", detoxify_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()
