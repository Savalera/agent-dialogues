"""Simulation config yaml loader."""

from pathlib import Path
from typing import Any

import yaml

from agentdialogues.exceptions import ConfigNotFoundError, YAMLParsingError


def load_simulation_config(path: str) -> dict[str, Any]:
    """Load a simulation scenario YAML from the given path.

    Args:
        path: Path to the YAML file, absolute or relative.

    Returns:
        Parsed YAML as a dictionary.

    Raises:
        ConfigNotFoundError: If the file does not exist.
        YAMLParsingError: If the file is not valid YAML.
    """
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise ConfigNotFoundError(f"Scenario file not found: '{config_path}'")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise YAMLParsingError(f"Failed to parse YAML: '{config_path}'") from e
