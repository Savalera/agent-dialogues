"""Simulation config utils."""

from pathlib import Path

import yaml


def load_simulation(simulation_id: str, config_dir: str = "src/sims") -> dict:
    """Load simulation config."""
    config_path = Path(config_dir) / f"{simulation_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found for: {simulation_id}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)
