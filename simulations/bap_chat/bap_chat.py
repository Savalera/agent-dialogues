"""Simulation module example.

Baby-Daddy simulation.
"""

from math import floor
from typing import Any, cast

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agentdialogues import (
    ChatAgent,
    ChatAgentState,
    Dialogue,
    DialogueItem,
    DialogueSimulationConfig,
    Roles,
    Simulation,
    convert_dialogue_to_chat_messages,
    create_chat_agent,
)


# === Simulation state ===
class SimulationState(BaseModel):
    """Simulation state."""

    dialogue: Dialogue
    seed_messages: list[DialogueItem]


# ⬇ Global runtime constants — set via init_simulation()
simulation_config: DialogueSimulationConfig
MAX_MESSAGES = 0
initiator: ChatAgent
responder: ChatAgent
initial_state: SimulationState


# === Graph nodes and edges ===
def call_initiator_node(state: SimulationState) -> dict[str, Any]:
    """Call initiator agent."""
    if len(state.seed_messages) > len(state.dialogue) / 2:
        return {
            "dialogue": state.dialogue
            + [state.seed_messages[floor(len(state.dialogue) / 2)]]
        }
    else:
        messages = convert_dialogue_to_chat_messages(
            human_role=Roles.RESPONDER, dialogue=state.dialogue
        )

        response = initiator.invoke(
            {
                "messages": messages,
                "system_prompt": simulation_config.initiator.system_prompt,
            }
        )

        r = ChatAgentState(**response)
        message = r.messages[-1]

        return {
            "dialogue": state.dialogue
            + [DialogueItem(role=Roles.INITIATOR, message=message)]
        }


def call_responder_node(state: SimulationState) -> dict[str, Any]:
    """Call responder agent."""
    messages = convert_dialogue_to_chat_messages(
        human_role=Roles.INITIATOR, dialogue=state.dialogue
    )

    response = responder.invoke(
        {
            "messages": messages,
            "system_prompt": simulation_config.responder.system_prompt,
        }
    )

    r = ChatAgentState(**response)
    message = r.messages[-1]

    return {
        "dialogue": state.dialogue
        + [DialogueItem(role=Roles.RESPONDER, message=message)]
    }


def should_continue(state: SimulationState) -> str:
    """Dialogue router."""
    dialogue = state.dialogue
    if len(dialogue) >= MAX_MESSAGES:
        return "end"
    else:
        return "continue"


# === Graph builder ===
workflow = StateGraph(SimulationState)
workflow.add_node("initiator", call_initiator_node)
workflow.add_node("responder", call_responder_node)

workflow.add_edge(START, "initiator")
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


# === Simulation creator ===
def create_simulation(config: dict[str, Any]) -> Simulation:
    """Initialize simulation."""
    global simulation_config, MAX_MESSAGES, initiator, responder, initial_state

    simulation_config = DialogueSimulationConfig(**config)

    MAX_MESSAGES = simulation_config.runtime.rounds * 2

    initiator = create_chat_agent(
        {
            "llm": cast(
                Runnable[list[BaseMessage], list[BaseMessage]],
                ChatOllama(model=simulation_config.initiator.model.model_name),
            )
        }
    )

    responder = create_chat_agent(
        {
            "llm": cast(
                Runnable[list[BaseMessage], list[BaseMessage]],
                ChatOllama(model=simulation_config.responder.model.model_name),
            )
        }
    )

    seed_messages = [
        DialogueItem(
            role=Roles.INITIATOR,
            message=HumanMessage(content=msg),
        )
        for msg in simulation_config.initiator.messages
    ]

    initial_state = SimulationState(
        dialogue=[],
        seed_messages=seed_messages,
    )

    return graph
