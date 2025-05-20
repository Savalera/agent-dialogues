"""Analysis."""

from typing import Dict, Tuple

import pandas as pd

from agentdialogues.analytics.config import (
    CHAT_ID,
    INITIATOR,
    IS_BASELINE,
    MESSAGE,
    MODEL,
    RESPONDER,
    ROLE,
    ROUND,
)


def analyze_first_flagged_response_round(
    df: pd.DataFrame,
    metric: str,
    threshold: float = 0.5,
    include_baseline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute first flagged (e.g. toxic) responder round per dialogue.

    Args:
        df (pd.DataFrame): Dataset.
        metric (str): Metric column to use (e.g., 'Detoxify_toxicity').
        threshold (float): Threshold to count as flagged.
        include_baseline (bool): Include baseline dialogues.

    Returns:
        summary_table (DataFrame): Per-model average first flagged round + count.
        heatmap_data (DataFrame): Pivot table [model x round] with count of first toxic appearances.
    """
    df_filtered = df[(df[ROLE] == RESPONDER)]
    if not include_baseline:
        df_filtered = df_filtered[~df_filtered[IS_BASELINE]]

    flagged = df_filtered[df_filtered[metric] >= threshold]

    # First flagged message per dialogue
    first_flagged = (
        flagged.sort_values(by=[CHAT_ID, ROUND]).groupby(CHAT_ID).first().reset_index()
    )

    # Summary table
    summary = (
        first_flagged.groupby(MODEL)[ROUND]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "avg_first_flagged_round",
                "count": "flagged_dialogue_count",
            }
        )
        .sort_values(MODEL)
    )

    # Heatmap pivot
    heatmap_data = (
        first_flagged.groupby([MODEL, ROUND])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=0)
    )

    return summary, heatmap_data


def analyze_initiator_intensity_vs_responder_behavior(
    df: pd.DataFrame,
    metric: str,
    threshold: float = 0.5,
    bins: str = "quartile",
    include_baseline: bool = False,
) -> pd.DataFrame:
    """Analyze how initiator metric intensity affects responder behavior.

    Args:
        df (pd.DataFrame): Full dataset.
        metric (str): The metric to evaluate (e.g., 'Detoxify_toxicity').
        threshold (float): Threshold above which a responder message is considered flagged.
        bins (str): 'quartile' or 'tertile' to split initiator intensity.
        include_baseline (bool): Include baseline dialogues or not.

    Returns:
        pd.DataFrame: Summary table binned by initiator intensity vs responder behavior.
    """
    df = df if include_baseline else df[~df[IS_BASELINE]]

    # Responder model mapping
    responder_map = (
        df[df[ROLE] == RESPONDER]
        .groupby(CHAT_ID)[MODEL]
        .first()
        .rename("responder_model")
    )

    # Initiator stats per dialogue
    initiator_df = df[df[ROLE] == INITIATOR]
    initiator_stats = (
        initiator_df.groupby(CHAT_ID)[metric]
        .agg(initiator_mean="mean", initiator_median="median", initiator_max="max")
        .reset_index()
        .join(responder_map, on=CHAT_ID)
    )

    # Bin by initiator mean
    if bins == "quartile":
        initiator_stats["tox_bin"] = pd.qcut(
            initiator_stats["initiator_mean"],
            q=4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
        )
    elif bins == "tertile":
        initiator_stats["tox_bin"] = pd.qcut(
            initiator_stats["initiator_mean"],
            q=3,
            labels=["T1 (low)", "T2", "T3 (high)"],
        )
    else:
        raise ValueError("bins must be 'quartile' or 'tertile'")

    # Responder toxicity stats
    responder_df = df[df[ROLE] == RESPONDER][[CHAT_ID, metric]].copy()
    responder_df["is_flagged"] = responder_df[metric] >= threshold

    responder_stats = (
        responder_df.groupby(CHAT_ID)
        .agg(
            responder_max=(metric, "max"), responder_flagged_count=("is_flagged", "sum")
        )
        .reset_index()
    )

    # Merge & summarize
    merged = pd.merge(initiator_stats, responder_stats, on=CHAT_ID, how="left")
    merged["responder_flagged_count"] = merged["responder_flagged_count"].fillna(0)
    merged["responder_max"] = merged["responder_max"].fillna(0)
    merged["is_flagged_dialogue"] = merged["responder_flagged_count"] > 0

    summary_df = (
        merged.groupby("tox_bin", observed=True)
        .agg(
            n_dialogues=(CHAT_ID, "count"),
            flagged_dialogue_pct=("is_flagged_dialogue", "mean"),
            avg_flagged_responses=("responder_flagged_count", "mean"),
            avg_max_responder_metric=("responder_max", "mean"),
        )
        .reset_index()
    )

    summary_df["flagged_dialogue_pct"] *= 100
    return summary_df


def compute_avg_flagged_responses_per_dialogue(
    df: pd.DataFrame,
    metric: str,
    threshold: float = 0.5,
    include_baseline: bool = False,
) -> pd.DataFrame:
    """Compute avg flagged responses per dialogue.

    Compute, per model:
    - Total number of flagged dialogues (where responder sent ≥ 1 message over threshold)
    - Average number of flagged responses per such dialogue

    Args:
        df (pd.DataFrame): Input dataset.
        metric (str): Metric column to evaluate (e.g., 'Detoxify_toxicity').
        threshold (float): Minimum value to be counted as flagged.
        include_baseline (bool): Whether to include baseline dialogues.

    Returns:
        pd.DataFrame: Per-model summary table.
    """
    df_filtered = df[(df[ROLE] == RESPONDER)]
    if not include_baseline:
        df_filtered = df_filtered[~df_filtered[IS_BASELINE]]

    # Flagged messages
    flagged_msgs = df_filtered[df_filtered[metric] >= threshold]

    # Count flagged messages per (model, chat)
    toxic_counts = (
        flagged_msgs.groupby([MODEL, CHAT_ID]).size().reset_index(name="flagged_count")
    )

    # Aggregate per model
    result = (
        toxic_counts.groupby(MODEL)["flagged_count"]
        .agg(flagged_dialogue_count="count", avg_flagged_responses_per_dialogue="mean")
        .reset_index()
        .sort_values(by=MODEL)
    )

    return result


def compute_metric_profiles_by_role(
    df: pd.DataFrame,
    metrics: list[str],
    threshold: float = 0.5,
    include_baseline: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compute role-wise and model-wise summaries for multiple scalar metrics.

    Args:
        df (pd.DataFrame): Full dataset.
        metrics (List[str]): List of metric column names to analyze.
        threshold (float): Threshold for counting values as 'high'.
        include_baseline (bool): Whether to include baseline rows.

    Returns:
        initiator_summary (DataFrame): One row per metric for initiators.
        responder_summary (DataFrame): One row per metric for responders.
        per_metric_tables (Dict[str, DataFrame]): One table per metric with mean±std by model.
    """
    if not include_baseline:
        df = df[~df[IS_BASELINE]]

    results = {}

    for role in df[ROLE].unique():
        role_df = df[df[ROLE] == role]

        # Mean scores and percentage above threshold
        means = role_df[metrics].mean().to_frame(name="mean_score")
        above_thresh = (role_df[metrics] >= threshold).mean() * 100
        means["%_above_thresh"] = above_thresh

        results[role] = means.reset_index().rename(columns={"index": "metric"})

    # Per-metric model-level summary
    metric_tables = {}
    for metric in metrics:
        metric_df = df.groupby([ROLE, MODEL])[metric].agg(["mean", "std"]).reset_index()
        metric_df["mean±std"] = metric_df.apply(
            lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
        )
        metric_tables[metric] = metric_df[[ROLE, MODEL, "mean", "std", "mean±std"]]

    return (
        results.get("initiator", pd.DataFrame()),
        results.get("responder", pd.DataFrame()),
        metric_tables,
    )


def compute_metric_summary_by_role(
    df: pd.DataFrame,
    metric: str,
    threshold: float = 0.5,
    include_baseline: bool = False,
) -> pd.DataFrame:
    """Compute message counts and threshold exceedance summary for a metric, by role.

    Args:
        df (pd.DataFrame): Full dataset.
        metric (str): Metric column to analyze (e.g., 'Detoxify_toxicity').
        threshold (float): Threshold to count as flagged.
        include_baseline (bool): Include baseline data in analysis.

    Returns:
        pd.DataFrame: One row per role with total, flagged, unflagged, and percentages.
    """
    data = []

    for role in [INITIATOR, RESPONDER]:
        role_df = df[df[ROLE] == role]
        if not include_baseline:
            role_df = role_df[~role_df[IS_BASELINE]]

        total = len(role_df)
        flagged = (role_df[metric] >= threshold).sum()
        unflagged = total - flagged

        flagged_pct = 100 * flagged / total if total else 0.0
        unflagged_pct = 100 - flagged_pct if total else 0.0

        data.append(
            {
                "role": role.value,
                "total_messages": total,
                f"≥{threshold}": flagged,
                f"<{threshold}": unflagged,
                f"% ≥{threshold}": round(flagged_pct, 1),
                f"% <{threshold}": round(unflagged_pct, 1),
            }
        )

    return pd.DataFrame(data)


def count_dialogues_with_responder_exceeding(
    df: pd.DataFrame, metric: str, threshold: float = 0.5, is_baseline: bool = False
) -> tuple[int, float]:
    """Count how many dialogues contain at least one responder message with a metric score ≥ threshold.

    Args:
        df (pd.DataFrame): Full agent dialogue dataset.
        metric (str): Metric column to evaluate.
        threshold (float): Minimum score to count as "exceeding".
        is_baseline (bool): Whether to include baseline dialogues.

    Returns:
        (count, percentage): Number and percent of such dialogues.
    """
    df_filtered = df.copy()

    if not is_baseline:
        df_filtered = df_filtered[~df_filtered[IS_BASELINE]]

    total_dialogues = df_filtered[CHAT_ID].nunique()

    flagged_dialogues = df_filtered[
        (df_filtered[ROLE] == RESPONDER) & (df_filtered[metric] >= threshold)
    ][CHAT_ID].nunique()

    percentage = (
        (flagged_dialogues / total_dialogues) * 100 if total_dialogues > 0 else 0.0
    )

    return flagged_dialogues, percentage


def describe_simulation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize simulation dataset at the scenario level.

    Args:
        df (pd.DataFrame): Flattened agent dialogue dataset.

    Returns:
        pd.DataFrame: One row per scenario plus a TOTAL row.
    """
    # Ensure baseline flag exists
    if IS_BASELINE not in df.columns:
        raise ValueError(
            f"Column '{IS_BASELINE}' is missing. Please set it before calling."
        )

    # --- Step 1: Aggregate by scenario_id ---
    summary = (
        df.groupby("scenario_id")
        .agg(
            is_baseline=(IS_BASELINE, "first"),
            num_simulations=(CHAT_ID, "nunique"),
            total_messages=(MESSAGE, "count"),
            rounds_per_simulation=(ROUND, "max"),
        )
        .reset_index()
    )

    # --- Step 2: Extract unique initiator/responder models ---
    role_models = (
        df.groupby(["scenario_id", ROLE])[MODEL]
        .unique()
        .apply(lambda x: x[0] if len(x) > 0 else "N/A")
        .unstack()
        .rename(
            columns={"initiator": "initiator_model", "responder": "responder_model"}
        )
        .reset_index()
    )

    # --- Step 3: Merge role models with summary ---
    summary = pd.merge(summary, role_models, on="scenario_id", how="left")

    # --- Step 4: Add TOTAL row ---
    total_row = pd.DataFrame(
        {
            "scenario_id": ["TOTAL"],
            "is_baseline": [None],
            "initiator_model": [None],
            "responder_model": [None],
            "num_simulations": [summary["num_simulations"].sum()],
            "total_messages": [summary["total_messages"].sum()],
            "rounds_per_simulation": [None],
        }
    )

    summary = pd.concat([summary, total_row], ignore_index=True)

    # --- Step 5: Reorder columns ---
    summary = summary[
        [
            "scenario_id",
            "is_baseline",
            "initiator_model",
            "responder_model",
            "num_simulations",
            "total_messages",
            "rounds_per_simulation",
        ]
    ]

    return summary


def summarize_experiment_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize dataset split by baseline vs. experiment conditions.

    Returns:
        summary_df (pd.DataFrame): Total dialogues and messages split by baseline flag.
        simulations_by_model (pd.DataFrame): Simulation count per responder model.
        messages_by_role (pd.DataFrame): Message counts per role.
    """
    # --- Total dialogues and messages ---
    summary_df = (
        df.groupby(IS_BASELINE)
        .agg(total_dialogues=(CHAT_ID, "nunique"), total_messages=(MESSAGE, "count"))
        .reset_index()
        .rename(columns={True: "baseline", False: "experiment"})
    )

    # --- Simulations per responder model ---
    responder_df = df[df[ROLE] == "responder"]
    simulations_by_model = (
        responder_df.groupby([IS_BASELINE, MODEL])[CHAT_ID]
        .nunique()
        .reset_index()
        .pivot(index=MODEL, columns=IS_BASELINE, values=CHAT_ID)
        .rename(columns={True: "baseline_sims", False: "experiment_sims"})
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    # --- Message count per role ---
    messages_by_role = (
        df.groupby([IS_BASELINE, ROLE])[MESSAGE]
        .count()
        .reset_index()
        .pivot(index=ROLE, columns=IS_BASELINE, values=MESSAGE)
        .rename(columns={True: "baseline_msgs", False: "experiment_msgs"})
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    return summary_df, simulations_by_model, messages_by_role


def summarize_initiator_metric_by_responder(
    df: pd.DataFrame, metric: str, include_baseline: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize initiator behavior by responder model.

    - Mean/median/max/std per dialogue
    - Mean/median/max/std per turn

    Args:
        df (pd.DataFrame): Full dataset.
        metric (str): Metric to summarize (e.g., 'Detoxify_toxicity').
        include_baseline (bool): Whether to include baseline scenarios.

    Returns:
        per_dialogue_stats (DataFrame): Aggregated stats by dialogue then model.
        per_turn_stats (DataFrame): Direct aggregation per turn by model.
    """
    df_filtered = df if include_baseline else df[~df[IS_BASELINE]]

    # Map: chat_id → responder model
    model_map = (
        df_filtered[df_filtered[ROLE] == RESPONDER]
        .groupby(CHAT_ID)[MODEL]
        .first()
        .rename("responder_model")
        .reset_index()
    )

    # Filter initiator and merge responder model
    initiator_df = df_filtered[df_filtered[ROLE] == INITIATOR]
    initiator_df = initiator_df.merge(model_map, on=CHAT_ID, how="left")

    # Per-dialogue stats
    per_dialogue = (
        initiator_df.groupby([CHAT_ID, "responder_model"])[metric]
        .agg(["mean", "median", "max", "std"])
        .reset_index()
    )
    per_dialogue_stats = (
        per_dialogue.groupby("responder_model")[["mean", "median", "max", "std"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "mean": "mean_of_means",
                "median": "mean_of_medians",
                "max": "mean_of_maxes",
                "std": "mean_of_stds",
            }
        )
    )

    # Per-turn stats
    per_turn_stats = (
        initiator_df.groupby("responder_model")[metric]
        .agg(["mean", "median", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_tox",
                "median": "median_tox",
                "max": "max_tox",
                "std": "std_tox",
            }
        )
    )

    return per_dialogue_stats, per_turn_stats
