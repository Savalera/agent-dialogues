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


class SimulationModuleError(SimulationError):
    """Raised when dynamic simulation loading fails."""


class YAMLParsingError(SimulationError):
    """Raised when a YAML file cannot be parsed."""


class DatasetError(Exception):
    """Base class for dataset-related errors."""


class DatasetDirectoryNotFoundError(DatasetError):
    """Raised when dataset directory is not found."""

    pass


class NoLogFilesFoundError(DatasetError):
    """Raised when no log files found."""

    pass


class LogProcessingError(DatasetError):
    """Raised when log processing fails."""

    pass


class EmptyDatasetError(DatasetError):
    """Raised when dataset has no data."""

    pass
