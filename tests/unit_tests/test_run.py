from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agents.simulation_agent import (
    ConversationItem,
    DialogueAgentConfig,
    SimulationState,
)
from domain import (
    DialogueInitiator,
    DialogueParticipant,
    Roles,
    Simulation,
    SimulationConfig,
)
from exceptions import SimulationExecutionError
from run import stream_simulation


@pytest.fixture
def fake_simulation():
    return Simulation(
        id="test",
        description="A test sim",
        initiator=DialogueInitiator(
            name="Alice",
            persona="team lead",
            model_name="mock-model",
            system_prompt="Be helpful",
            initial_message="Hello!",
        ),
        responder=DialogueParticipant(
            name="Bob",
            persona="team member",
            model_name="mock-model",
            system_prompt="Be clear",
        ),
        config=SimulationConfig(rounds=1),
    )


@patch("run.app.stream")
@patch("run.ChatOllama")
def test_stream_simulation_success(mock_ollama, mock_stream, fake_simulation):
    mock_ollama.return_value = MagicMock(spec=Runnable)

    # Build the expected state manually
    base_state = SimulationState(
        conversation=[
            ConversationItem(
                role=Roles.INITIATOR, message=HumanMessage(content="Hello!")
            )
        ],
        MAX_MESSAGES=2,
        initiator=DialogueAgentConfig(
            llm=MagicMock(spec=Runnable), system_prompt="Initiator prompt."
        ),
        responder=DialogueAgentConfig(
            llm=MagicMock(spec=Runnable), system_prompt="Responder prompt."
        ),
    )

    chunk1 = base_state.model_copy(
        update={
            "conversation": [
                ConversationItem(
                    role=Roles.INITIATOR, message=HumanMessage(content="Hello!")
                )
            ]
        }
    )

    chunk2 = base_state.model_copy(
        update={
            "conversation": [
                ConversationItem(role=Roles.RESPONDER, message=AIMessage(content="Hi!"))
            ]
        }
    )

    mock_stream.return_value = iter([chunk1.model_dump(), chunk2.model_dump()])

    results = list(stream_simulation(fake_simulation))

    assert len(results) == 2
    assert results[0].conversation[0].message.content == "Hello!"
    assert results[1].conversation[0].message.content == "Hi!"


@patch("run.app.stream")
@patch("run.ChatOllama")
def test_stream_simulation_failure(mock_ollama, mock_stream, fake_simulation):
    mock_ollama.side_effect = RuntimeError("LLM unavailable")

    with pytest.raises(SimulationExecutionError, match="Simulation execution failed."):
        list(stream_simulation(fake_simulation))
