"""Simulation dataset analytics config."""

from agentdialogues.core.base import Roles

# General column names
SCENARIO_ID = "scenario_id"
CHAT_ID = "chat_id"
ROUND = "round"
ROLE = "role"
MODEL = "model_name"
MESSAGE = "message"
IS_BASELINE = "is_baseline"

# Role values
INITIATOR = Roles.INITIATOR
RESPONDER = Roles.RESPONDER

# Default metrics (can be overridden in function calls)
DEFAULT_METRICS = [
    "Detoxify_toxicity",
    "Detoxify_severe_toxicity",
    "Detoxify_obscene",
    "Detoxify_threat",
    "Detoxify_insult",
    "Detoxify_identity_attack",
    "Detoxify_sexual_explicit",
]
