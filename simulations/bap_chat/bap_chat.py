"""Simulation module example.

Chat dialogue with two participants.
"""

from math import floor
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, PositiveInt

from agentdialogues import (
    Dialogue,
    DialogueItem,
    DialogueSimulationConfig,
    Roles,
    chat_agent,
    convert_dialogue_to_chat_messages,
)


# === Simulation state ===
class Runtime(BaseModel):
    """Simulation runtime configuration."""

    MAX_MESSAGES: PositiveInt


class SimulationState(BaseModel):
    """Simulation state."""

    dialogue: Dialogue = []
    raw_config: dict[str, Any] = Field(default_factory=dict)
    config: Optional[DialogueSimulationConfig] = None
    runtime: Optional[Runtime] = None


# === Graph nodes and edges ===
def setup_node(state: SimulationState) -> dict[str, Any]:
    """Set up sumulation state."""
    config = DialogueSimulationConfig(**state.raw_config)

    MAX_MESSAGES = config.runtime.rounds * 2

    return {"config": config, "runtime": {"MAX_MESSAGES": MAX_MESSAGES}}


def call_initiator_node(state: SimulationState) -> dict[str, Any]:
    """Call initiator agent."""
    assert state.config
    assert state.config.initiator.messages

    seed_messages = state.config.initiator.messages

    if len(seed_messages) > len(state.dialogue) / 2:
        return {
            "dialogue": state.dialogue
            + [
                DialogueItem(
                    role=Roles.INITIATOR,
                    message=seed_messages[floor(len(state.dialogue) / 2)],
                )
            ]
        }
    else:
        messages = convert_dialogue_to_chat_messages(
            human_role=Roles.RESPONDER, dialogue=state.dialogue
        )

        response = chat_agent.invoke(
            {
                "messages": messages,
                "system_prompt": state.config.initiator.system_prompt,
                "raw_config": {
                    "model_name": state.config.initiator.model.model_name,
                    "provider": state.config.initiator.model.provider,
                    "seed": state.config.seed,
                },
            }
        )

        message = response["messages"][-1]

        return {
            "dialogue": state.dialogue
            + [DialogueItem(role=Roles.INITIATOR, message=message.content)]
        }


def call_responder_node(state: SimulationState) -> dict[str, Any]:
    """Call responder agent."""
    assert state.config

    messages = convert_dialogue_to_chat_messages(
        human_role=Roles.INITIATOR, dialogue=state.dialogue
    )

    response = chat_agent.invoke(
        {
            "messages": messages,
            "system_prompt": state.config.responder.system_prompt,
            "raw_config": {
                "model_name": state.config.responder.model.model_name,
                "provider": state.config.responder.model.provider,
                "seed": state.config.seed,
            },
        }
    )

    message = response["messages"][-1]

    return {
        "dialogue": state.dialogue
        + [DialogueItem(role=Roles.RESPONDER, message=message.content)]
    }


def should_continue(state: SimulationState) -> str:
    """Dialogue router."""
    dialogue = state.dialogue

    assert state.runtime

    if len(dialogue) >= state.runtime.MAX_MESSAGES:
        return "end"
    else:
        return "continue"


# === Graph builder ===
workflow = StateGraph(SimulationState)
workflow.add_node("setup", setup_node)
workflow.add_node("initiator", call_initiator_node)
workflow.add_node("responder", call_responder_node)

workflow.add_edge(START, "setup")
workflow.add_edge("setup", "initiator")
workflow.add_conditional_edges(
    "responder",
    should_continue,
    {
        "end": END,
        "continue": "initiator",
    },
)
workflow.add_edge("initiator", "responder")
graph = workflow.compile()
