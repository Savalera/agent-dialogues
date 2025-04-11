"""Agent Dialogues module."""

from .agents.chat_agent import graph as chat_agent
from .agents.detoxify_agent import graph as detoxify_agent
from .analytics import build_simulation_dataset_from_manifest
from .core.base import (
    BaseSimulationConfig,
    ChatProviders,
    Dialogue,
    DialogueItem,
    DialogueParticipantConfig,
    DialogueParticipantWithMessagesConfig,
    DialogueSimulationConfig,
    Roles,
)
from .core.bootstrap import bootstrap_simulation
from .core.utils import convert_dialogue_to_chat_messages
from .exceptions import SimulationExecutionError

__all__ = [
    "BaseSimulationConfig",
    "ChatProviders",
    "Dialogue",
    "DialogueItem",
    "DialogueParticipantConfig",
    "DialogueParticipantWithMessagesConfig",
    "DialogueSimulationConfig",
    "Roles",
    "SimulationExecutionError",
    "bootstrap_simulation",
    "build_simulation_dataset_from_manifest",
    "chat_agent",
    "convert_dialogue_to_chat_messages",
    "detoxify_agent",
]
