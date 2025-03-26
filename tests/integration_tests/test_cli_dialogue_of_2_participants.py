import builtins
import os
import pathlib
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from sim_cli import main


@pytest.fixture
def isolated_logs_dir(monkeypatch):
    """Redirect Path.open when writing to logs directory."""
    test_logs = tempfile.mkdtemp(prefix="test_logs_")

    def safe_open(path_obj, *args, **kwargs):
        if "logs" in str(path_obj):
            filename = path_obj.name
            redirected_path = os.path.join(test_logs, filename)
            return builtins.open(redirected_path, *args, **kwargs)
        # fallback to original behavior
        return builtins.open(path_obj, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", safe_open)

    yield test_logs
    shutil.rmtree(test_logs)


@patch("run.ChatOllama")
def test_cli_main_integration(mock_chatollama, isolated_logs_dir):
    mock_instance = MagicMock(spec=Runnable)
    mock_instance.invoke.return_value = AIMessage(content="Mocked response")
    mock_chatollama.return_value = mock_instance

    from sys import argv

    argv[:] = ["prog", "--sim", "baby-daddy"]

    main()

    logs = os.listdir(isolated_logs_dir)
    assert len(logs) == 1
