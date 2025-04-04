"""Agent Dialogues domain objects."""

from enum import Enum

from langchain_core.messages import AnyMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, PositiveInt

Simulation = CompiledStateGraph


# === Constants ===
class Roles(str, Enum):
    """Defines the roles that a dialogue agent can take during a simulation.

    Attributes:
        INITIATOR: The agent who starts the dialogue.
        RESPONDER: The agent who responds to the initiator.

    Example:
        >>> role = Roles.INITIATOR
        >>> print(role.value)
        'initiator'
    """

    INITIATOR = "initiator"
    RESPONDER = "responder"


# === Domain ===
class DialogueItem(BaseModel):
    """Dialogue item."""

    role: Roles
    message: AnyMessage

    model_config = ConfigDict(arbitrary_types_allowed=True)


Dialogue = list[DialogueItem]


# === Simulation config schema ===
class BaseSimulationConfig(BaseModel):
    """Base simulation definition schema.

    Extend this schema to define your simulation.

    Use the schema to validate your simulation yaml file.

    Attributes:
        id: Unique simulation ID.
        name: Name for display and logs.
        rounds: Number of dialogue rounds (must be > 0).
    """

    id: str
    name: str
    rounds: PositiveInt


class DialogueParticipantConfig(BaseModel):
    """Represents an agent participating in a dialogue.

    Attributes:
        name: Display name of the agent.
        persona: Short description of the agent's behavior or perspective.
        model_name: Name of the LLM model this agent uses.
        system_prompt: System message used to steer the agent's responses.
    """

    name: str
    persona: str
    model_name: str
    system_prompt: str


class DialogueInitiatorConfig(DialogueParticipantConfig):
    """A dialogue participant that initiates the dialogue.

    Inherits from:
        DialogueParticipant

    Adds:
        initial_message: The message that starts the dialogue.
    """

    initial_message: str


class DialogueSimulationConfig(BaseSimulationConfig):
    """Defines a full simulation schema including both dialogue participants.

    Inherits from:
        BaseSimulationSchema – includes simulation-level metadata like `id`, `title`, and `rounds`.

    Attributes:
        initiator: The agent who initiates the dialogue, including their initial message.
        responder: The agent who responds to the initiator.
    """

    initiator: DialogueInitiatorConfig
    responder: DialogueParticipantConfig
