"""Exceptions."""


class SimulationError(Exception):
    """General simulation-related error."""


class ArgumentParsingError(SimulationError):
    """Raised when CLI argument parsing fails."""


class CLIRuntimeError(SimulationError):
    """Raised when a CLI runtime error occurs."""


class ConfigNotFoundError(SimulationError):
    """Raised when the config file for a simulation is not found."""


class DialogueAgentError(SimulationError):
    """Raised when calling the dialogue agent fails."""


class LLMInvocationError(SimulationError):
    """Raised when an LLM invocation fails."""


class SimulationExecutionError(SimulationError):
    """Raised when simulation execution fails (e.g. model config or runtime error)."""


class YAMLParsingError(SimulationError):
    """Raised when a YAML file cannot be parsed."""
