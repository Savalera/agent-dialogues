import tempfile
from pathlib import Path

import pytest

from agentdialogues.exceptions import ConfigNotFoundError, YAMLParsingError
from agentdialogues.utils.yaml_loader import load_simulation_config


def test_load_simulation_config_success():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        f.write("name: test\nrounds: 5")
        temp_path = f.name

    result = load_simulation_config(temp_path)
    assert result == {"name": "test", "rounds": 5}

    Path(temp_path).unlink()


def test_load_simulation_config_file_not_found():
    with pytest.raises(ConfigNotFoundError):
        load_simulation_config("non_existent_file.yaml")


def test_load_simulation_config_invalid_yaml():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        f.write("invalid: [unbalanced brackets")
        temp_path = f.name

    with pytest.raises(YAMLParsingError):
        load_simulation_config(temp_path)

    Path(temp_path).unlink()
