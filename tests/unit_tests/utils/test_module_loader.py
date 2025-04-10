import tempfile
from pathlib import Path

import pytest

from agentdialogues.exceptions import SimulationModuleError
from agentdialogues.utils.module_loader import load_simulation_module


def test_load_valid_simulation_module():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_code = "graph = 'fake-graph'"
        sim_path = Path(temp_dir) / "test_sim.py"
        sim_path.write_text(sim_code)

        module = load_simulation_module(str(sim_path))
        assert hasattr(module, "graph")
        assert module.graph == "fake-graph"


def test_load_module_file_not_found():
    with pytest.raises(SimulationModuleError) as exc_info:
        load_simulation_module("non_existent_sim.py", default_dir=".")

    assert "Simulation file not found" in str(exc_info.value)


def test_load_module_invalid_python():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_path = Path(temp_dir) / "broken_sim.py"
        sim_path.write_text("def invalid_python(:)")

        with pytest.raises(SimulationModuleError) as exc_info:
            load_simulation_module(str(sim_path))

        assert "Failed to execute module" in str(exc_info.value)
