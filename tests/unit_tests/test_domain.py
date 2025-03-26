import pytest
from pydantic import ValidationError

from domain import (
    DialogueInitiator,
    DialogueParticipant,
    Roles,
    Simulation,
    SimulationConfig,
)


def test_roles_enum():
    assert Roles.INITIATOR.value == "initiator"
    assert Roles.RESPONDER.value == "responder"


def test_dialogue_participant_creation():
    participant = DialogueParticipant(
        name="Test Agent",
        persona="Friendly assistant",
        model_name="test-model",
        system_prompt="You are helpful.",
    )

    assert participant.name == "Test Agent"
    assert participant.persona == "Friendly assistant"
    assert participant.model_name == "test-model"
    assert participant.system_prompt == "You are helpful."


def test_dialogue_initiator_creation():
    initiator = DialogueInitiator(
        name="Initiator Agent",
        persona="Confident leader",
        model_name="initiator-model",
        system_prompt="You start conversations proactively.",
        initial_message="Hello there!",
    )

    assert initiator.name == "Initiator Agent"
    assert initiator.initial_message == "Hello there!"


def test_simulation_config_creation():
    config = SimulationConfig(rounds=5)
    assert config.rounds == 5


def test_simulation_creation():
    initiator = DialogueInitiator(
        name="Initiator",
        persona="Friendly initiator",
        model_name="initiator-model",
        system_prompt="Initiate clearly.",
        initial_message="Start conversation.",
    )
    responder = DialogueParticipant(
        name="Responder",
        persona="Thoughtful responder",
        model_name="responder-model",
        system_prompt="Respond thoughtfully.",
    )
    config = SimulationConfig(rounds=10)

    sim = Simulation(
        id="test-simulation",
        description="A simple test simulation",
        initiator=initiator,
        responder=responder,
        config=config,
    )

    assert sim.id == "test-simulation"
    assert sim.config.rounds == 10
    assert sim.initiator.initial_message == "Start conversation."


def test_dialogue_participant_missing_fields():
    with pytest.raises(ValidationError):
        DialogueParticipant(name="Missing fields")  # type: ignore


def test_simulation_negative_rounds():
    with pytest.raises(ValidationError):
        SimulationConfig(rounds=-1)
