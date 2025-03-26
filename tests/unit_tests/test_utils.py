import tempfile
from pathlib import Path

import pytest

from exceptions import ConfigNotFoundError, YAMLParsingError
from utils import load_simulation


def test_load_simulation_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        sim_file = Path(tmpdir) / "test.yaml"
        sim_file.write_text("id: test\nconfig:\n  rounds: 3")

        result = load_simulation("test", config_dir=tmpdir)

        assert isinstance(result, dict)
        assert result["id"] == "test"
        assert result["config"]["rounds"] == 3


def test_load_simulation_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(
            ConfigNotFoundError, match="No config file found for: `missing-sim`"
        ):
            load_simulation("missing-sim", config_dir=tmpdir)


def test_load_simulation_invalid_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        sim_file = Path(tmpdir) / "bad.yaml"
        sim_file.write_text("config: [unclosed_list")

        with pytest.raises(YAMLParsingError, match="YAML parsing failed"):
            load_simulation("bad", config_dir=tmpdir)
