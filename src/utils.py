"""Simulation config utils."""

from pathlib import Path

import yaml

from exceptions import ConfigNotFoundError, YAMLParsingError


def load_simulation(simulation_id: str, config_dir: str = "sims") -> dict:
    """Load simulation config."""
    config_path = Path(config_dir) / f"{simulation_id}.yaml"

    if not config_path.exists():
        raise ConfigNotFoundError(f"No config file found for: `{simulation_id}`")

    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise YAMLParsingError(f"YAML parsing failed for `{simulation_id}`") from e
