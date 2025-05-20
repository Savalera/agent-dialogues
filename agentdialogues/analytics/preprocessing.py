"""Analyitics preprocessing."""

import pandas as pd

from agentdialogues.analytics.config import IS_BASELINE, MODEL, SCENARIO_ID


def load_dialogue_dataset(
    path,
    model_name_map=None,
    baseline_marker="-bl-",
):
    """Load a simulation dataset CSV and perform lightweight preprocessing.

    Args:
        path (str): Path to the CSV file.
        model_name_map (dict): Optional mapping to normalize model names.
        baseline_marker (str): Substring that identifies baseline rows.
        scenario_column (str): Column name containing scenario identifiers.

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
