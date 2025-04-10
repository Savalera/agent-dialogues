"""This module bootstraps the simulation and returns what you need to run it."""

from typing import Any, Tuple

from langgraph.graph.state import CompiledStateGraph

from agentdialogues.exceptions import SimulationExecutionError
from agentdialogues.utils import load_simulation_config, load_simulation_module


def bootstrap_simulation(
    simulation_module_path: str, simulation_config_path: str
) -> Tuple[CompiledStateGraph, dict[str, Any]]:
    """Bootstrap simulation runtime."""
    try:

        simulation = load_simulation_module(simulation_module_path)
        simulation_config = load_simulation_config(simulation_config_path)

        return simulation.graph, simulation_config

    except Exception as e:
        raise SimulationExecutionError("Simulation bootstrap failed.") from e
