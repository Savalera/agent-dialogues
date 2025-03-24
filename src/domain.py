"""Domain definitions."""

from enum import Enum

from pydantic import BaseModel


class Roles(str, Enum):
    """Role constants."""

    INITIATOR = "initiator"
    RESPONDER = "responder"


class DialogueParticipant(BaseModel):
    """Participant in a simulated dialogue."""

    name: str
    persona: str
    model_name: str
    system_prompt: str


class DialogueInitiator(DialogueParticipant):
    """Initiating party of the dialogue."""

    initial_message: str


class SimulationConfig(BaseModel):
    """Configuration for the simulation."""

    rounds: int


class Simulation(BaseModel):
    """Simulation definition."""

    id: str
    description: str
    initiator: DialogueInitiator
    responder: DialogueParticipant
    config: SimulationConfig
