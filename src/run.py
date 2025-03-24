"""Command Line runner."""

from typing import Generator

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agents import simulation_agent as app
from agents.simulation_agent import (
    ConversationItem,
    DialogueAgentConfig,
    SimulationState,
)
from domain import Roles, Simulation


def stream_simulation(sim: Simulation) -> Generator[SimulationState, None, None]:
    """Yield each step of streamed simulation."""
    initiator = DialogueAgentConfig(
        llm=ChatOllama(model=sim.initiator.model_name),
        system_prompt=sim.initiator.system_prompt,
    )

    responder = DialogueAgentConfig(
        llm=ChatOllama(model=sim.responder.model_name),
        system_prompt=sim.responder.system_prompt,
    )

    conversation = [
        ConversationItem(
            role=Roles.INITIATOR,
            message=HumanMessage(content=sim.initiator.initial_message),
        )
    ]

    initial_state = SimulationState(
        conversation=conversation,
        MAX_MESSAGES=sim.config.rounds * 2,
        initiator=initiator,
        responder=responder,
    )

    for chunk in app.stream(
        initial_state,
        stream_mode="values",
    ):
        yield SimulationState(**chunk)
