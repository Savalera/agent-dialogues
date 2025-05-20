"""Dataset builder."""

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from agentdialogues.exceptions import (
    DatasetDirectoryNotFoundError,
    EmptyDatasetError,
    LogProcessingError,
    NoLogFilesFoundError,
)


def flatten_dialogue(
    simulation_id: str, chat_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Flatten dialogue and compute derived fields."""
    scenario_id = chat_data.get("simulation_config", {}).get("id", "unknown_scenario")
    chat_id = chat_data.get("chat_id", "unknown_chat")
    batch_id = chat_data.get("batch_id", "unknown_batch")
    seed = chat_data.get("seed", "unknown_seed")
    dialogue = chat_data.get("dialogue", [])

    rows = []
    for idx, msg in enumerate(dialogue):
        base = {
            "simulation_id": simulation_id,
            "scenario_id": scenario_id,
            "chat_id": chat_id,
            "batch_id": batch_id,
            "seed": seed,
            "round": idx // 2 + 1,
            "message_index": idx,
            "role": msg.get("role"),
            "name": msg.get("name"),
            "model_name": chat_data.get("simulation_config", {})
            .get(msg.get("role"), {})
            .get("model", {})
            .get("model_name"),
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


def build_simulation_dataset_from_manifest(
    simulation_id: str,
    scenario_ids: Sequence[str],
    logs_dir: Path,
    datasets_dir: Path,
    save_scenario_datasets: bool = False,
) -> Path:
    """Build one dataset for a simulation by combining logs of multiple scenarios."""
    all_rows = []
    simulation_dataset_dir = datasets_dir / simulation_id
    simulation_dataset_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id in scenario_ids:
        scenario_log_dir = logs_dir / scenario_id
        if not scenario_log_dir.exists():
            raise DatasetDirectoryNotFoundError(
                f"Missing log dir for scenario: {scenario_id}"
            )

        json_files = list(scenario_log_dir.glob("*.json"))
        if not json_files:
            raise NoLogFilesFoundError(f"No .json files found in: {scenario_log_dir}")

        scenario_rows = []
        for file_path in json_files:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    chat_log = json.load(f)
                    flat_rows = flatten_dialogue(simulation_id, chat_log)
                    scenario_rows.extend(flat_rows)
            except Exception as e:
                raise LogProcessingError(
                    f"Failed to process {file_path.name}: {e}"
                ) from e

        if not scenario_rows:
            raise EmptyDatasetError(f"No data found in scenario: {scenario_id}")

        all_rows.extend(scenario_rows)

        if save_scenario_datasets:
            scenario_df = pd.DataFrame(scenario_rows)
            scenario_file = simulation_dataset_dir / f"aggregated_{scenario_id}.csv"
            scenario_df.to_csv(scenario_file, index=False)

    if not all_rows:
        raise EmptyDatasetError(f"No messages found in any of: {scenario_ids}")

    simulation_df = pd.DataFrame(all_rows)
    simulation_output_file = simulation_dataset_dir / f"aggregated_{simulation_id}.csv"
    simulation_df.to_csv(simulation_output_file, index=False)
    return simulation_output_file
