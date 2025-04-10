import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd

from agentdialogues.utils.dataset import flatten_dialogue, prepare_simulation_dataset


def test_flatten_dialogue_with_evaluation():
    fake_chat = {
        "chat_id": "abc123",
        "batch_id": "batch-001",
        "seed": 42,
        "dialogue": [
            {
                "role": "initiator",
                "name": "Alice",
                "message": "Hi there!",
                "meta": [
                    {
                        "evaluator": "Detoxify",
                        "score": {
                            "toxicity": 0.5,
                            "insult": 0.1,
                        },
                    }
                ],
            },
            {
                "role": "responder",
                "name": "Bob",
                "message": "Hello!",
                "meta": [],
            },
        ],
    }

    rows = flatten_dialogue(fake_chat)

    assert len(rows) == 2
    assert rows[0]["chat_id"] == "abc123"
    assert rows[0]["Detoxify_toxicity"] == 0.5
    assert rows[1]["message"] == "Hello!"
    assert "Detoxify_toxicity" not in rows[1]


@patch("agentdialogues.utils.dataset.console.print")
@patch("agentdialogues.utils.dataset.track")
def test_prepare_simulation_dataset_happy_path(mock_track, mock_print):
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        logs_path = tmp_path / "test_batch"
        logs_path.mkdir()

        # Create a fake log file
        data = {
            "chat_id": "log-1",
            "batch_id": "test_batch",
            "seed": 123,
            "dialogue": [
                {
                    "role": "initiator",
                    "name": "Alice",
                    "message": "Hello",
                    "meta": [{"evaluator": "Detoxify", "score": {"toxicity": 0.1}}],
                }
            ],
        }
        log_file = logs_path / "log_1.json"
        with log_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        # Simulate the progress track
        mock_track.return_value = [log_file]

        prepare_simulation_dataset("test_batch", logs_base_path=tmp_path)

        output_file = logs_path / "aggregated_scores.csv"
        assert output_file.exists()

        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert df["Detoxify_toxicity"].iloc[0] == 0.1
