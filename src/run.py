"""Command Line runner."""

from typing import Generator, cast

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables.base import Runnable
from langchain_ollama import ChatOllama

from agents.dialogue_agent import DialogueAgentConfig
from agents.simulation_agent import (
    ConversationItem,
    SimulationState,
)
from agents.simulation_agent import graph as app
from domain import Roles, Simulation
from exceptions import SimulationExecutionError


def stream_simulation(sim: Simulation) -> Generator[SimulationState, None, None]:
    """Yield each step of streamed simulation."""
    try:
        initiator = DialogueAgentConfig(
            llm=cast(
                Runnable[list[BaseMessage], list[BaseMessage]],
                ChatOllama(model=sim.initiator.model_name),
            ),
            system_prompt=sim.initiator.system_prompt,
        )

        responder = DialogueAgentConfig(
            llm=cast(
                Runnable[list[BaseMessage], list[BaseMessage]],
                ChatOllama(model=sim.responder.model_name),
            ),
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

    except Exception as e:
        raise SimulationExecutionError("Simulation execution failed.") from e
