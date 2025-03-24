"""Constants."""

from enum import Enum


class Roles(str, Enum):
    """Role constants."""

    INITIATOR = "initiator"
    RESPONDER = "responder"
