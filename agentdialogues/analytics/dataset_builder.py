"""Dataset builder."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agentdialogues.exceptions import (
    DatasetDirectoryNotFoundError,
    EmptyDatasetError,
    LogProcessingError,
    NoLogFilesFoundError,
)


def flatten_dialogue(chat_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten dialogue and compute derived fields."""
    scenario_id = chat_data.get("simulation_config", "unknown scenario").get(
        "id", "unknown scenario"
    )
    chat_id = chat_data.get("chat_id", "unknown_chat")
    batch_id = chat_data.get("batch_id", "unknown_batch")
    seed = chat_data.get("seed", "unknown_seed")
    dialogue = chat_data.get("dialogue", [])

    rows = []
    for idx, msg in enumerate(dialogue):
        base = {
            "scenario_id": scenario_id,
            "chat_id": chat_id,
            "batch_id": batch_id,
            "seed": seed,
            "round": idx // 2 + 1,
            "message_index": idx,
            "role": msg.get("role"),
            "name": msg.get("name"),
            "message": msg.get("message"),
            "message_length": len(msg.get("message", "")),
        }

        for evaluator in msg.get("meta", []):
            ev_name = evaluator.get("evaluator", "unknown")
            score_dict = evaluator.get("score", {})
            for k, v in score_dict.items():
                base[f"{ev_name}_{k}"] = v

        rows.append(base)

    return rows


def prepare_simulation_dataset(
    scenario_id: str, logs_base_path: str = "logs", output_filename: str | None = None
) -> Path:
    """Prepare aggregated CSV dataset from JSON logs."""
    input_dir = Path(logs_base_path) / scenario_id
    if output_filename is None:
        output_filename = f"aggregated_{scenario_id}.csv"
    output_file = input_dir / output_filename

    if not input_dir.exists():
        raise DatasetDirectoryNotFoundError(f"Directory not found: {input_dir}")

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise NoLogFilesFoundError(f"No .json files found in: {input_dir}")

    all_rows = []
    for file_path in json_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                chat_log = json.load(f)
                flat_rows = flatten_dialogue(chat_log)
                all_rows.extend(flat_rows)
        except Exception as e:
            raise LogProcessingError(f"Failed to process {file_path.name}: {e}") from e

    if not all_rows:
        raise EmptyDatasetError("No messages extracted. Aborting.")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)
    return output_file
