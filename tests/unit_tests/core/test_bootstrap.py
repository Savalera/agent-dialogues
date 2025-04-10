import pytest
from pydantic import BaseModel

from agentdialogues import SimulationExecutionError, bootstrap_simulation


# === Setup mocks ===
class DummyState(BaseModel):
    foo: str = "bar"


@pytest.fixture
def dummy_simulation(tmp_path) -> tuple[str, str]:
    """Create a dummy simulation module and config YAML."""
    # Simulation module
    sim_code = """
from langgraph.graph import StateGraph
from pydantic import BaseModel

class DummyState(BaseModel):
    foo: str = "bar"

workflow = StateGraph(DummyState)
workflow.add_node("start", lambda state: {"foo": "baz"})
workflow.set_entry_point("start")
graph = workflow.compile()
"""

    sim_path = tmp_path / "dummy_sim.py"
    sim_path.write_text(sim_code)

    # YAML config file
    config = "foo: baz\n"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config)

    return str(sim_path), str(config_path)


# === Tests ===
def test_bootstrap_success(dummy_simulation):
    sim_path, config_path = dummy_simulation
    graph, config = bootstrap_simulation(sim_path, config_path)

    assert hasattr(graph, "invoke")
    assert isinstance(config, dict)
    assert config["foo"] == "baz"


def test_bootstrap_failure_invalid_module():
    with pytest.raises(SimulationExecutionError):
        bootstrap_simulation("not_a_real_path.py", "also_missing.yaml")
