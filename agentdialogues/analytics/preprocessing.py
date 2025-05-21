"""Analytics preprocessing."""

from typing import Optional

import pandas as pd

from agentdialogues.analytics.config import IS_BASELINE, MODEL, SCENARIO_ID


def load_dialogue_dataset(
    path: str,
    model_name_map: Optional[dict[str, str]] = None,
    baseline_marker: str = "-bl-",
) -> pd.DataFrame:
    """Load a simulation dataset CSV and perform lightweight preprocessing.

    Args:
        path (str): Path to the CSV file.
        model_name_map (Optional[dict[str, str]]): Optional mapping to normalize model names.
        baseline_marker (str): Substring that identifies baseline rows.

    Returns:
        pd.DataFrame: Cleaned dataset with standard fields.
    """
    df = pd.read_csv(path)

    # Baseline detection
    df[IS_BASELINE] = df[SCENARIO_ID].str.contains(baseline_marker)

    # Standardize model names
    if model_name_map:
        df[MODEL] = df[MODEL].replace(model_name_map)

    return df
