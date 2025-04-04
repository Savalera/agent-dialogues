"""Simulation module loader."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from agentdialogues.exceptions import SimulationModuleError


def load_simulation_module(path: str, default_dir: str = "simulations") -> ModuleType:
    """Dynamically load a simulation module from file.

    If `path` is just a filename, it looks in the default 'simulations/' directory.
    Otherwise, uses the given path as-is (absolute or relative).

    Args:
        path: Filename or full path to the simulation `.py` file.
        default_dir: Default directory to look in if only filename is given.

    Returns:
        The loaded module object.

    Raises:
        SimulationModuleError: If loading fails.
    """
    path_obj = Path(path)

    # If just a filename, look in default directory
    if not path_obj.parent or path_obj.parent == Path("."):
        path_obj = Path(default_dir) / path_obj

    path_obj = path_obj.expanduser().resolve()

    if not path_obj.exists():
        raise SimulationModuleError(f"Simulation file not found: '{path_obj}'")

    module_name = path_obj.stem
    spec = importlib.util.spec_from_file_location(module_name, path_obj)
    if spec is None or spec.loader is None:
        raise SimulationModuleError(f"Could not load module from: '{path_obj}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise SimulationModuleError(f"Failed to execute module: '{path_obj}'") from e

    return module
