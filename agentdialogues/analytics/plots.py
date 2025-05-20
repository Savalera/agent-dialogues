"""Plots."""

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from agentdialogues.analytics.config import (
    CHAT_ID,
    INITIATOR,
    IS_BASELINE,
    MODEL,
    RESPONDER,
    ROLE,
    ROUND,
)


def plot_dialogue_counts_above_threshold(
    df: pd.DataFrame,
    metrics: list[str],
    threshold: float = 0.5,
    fixed_ylim: int = 10,
    include_baseline: bool = False,
    role: str = RESPONDER,
    ncols: int = 4,
) -> None:
    """Plot a grid of bar charts showing how many dialogues contain at least one message from the given role with a metric above threshold.

    Args:
        df (pd.DataFrame): Input dataset.
        metrics (list[str]): List of metric columns to plot.
        threshold (float): Threshold value for each metric.
        fixed_ylim (int): Max y-axis value for uniform chart scaling.
        include_baseline (bool): Whether to include baseline data.
        role (str): Which role to analyze (default: responder).
    """
    # Filter
    df_filtered = df if include_baseline else df[~df[IS_BASELINE]]
    df_role = df_filtered[df_filtered[ROLE] == role]
    model_names = sorted(df_role[MODEL].unique())

    n_metrics = len(metrics)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Dialogues with at least one message ≥ threshold
        flagged = df_role[df_role[metric] >= threshold]
        counts = flagged.groupby(MODEL)[CHAT_ID].nunique()

        # Ensure all models are shown, even with 0
        counts_full = pd.Series(0, index=model_names)
        counts_full.update(counts)

        # Plot
        counts_full.plot(kind="bar", ax=ax)
        ax.set_title(f"{metric}", fontsize=10)
        ax.set_xlabel("Model")
        ax.set_ylabel("Dialogue Count")
        ax.set_ylim(0, fixed_ylim)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
        ax.grid(True, linestyle="--", linewidth=0.5, axis="x")

    # Remove unused axes
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_first_flagged_round_heatmap(
    heatmap_data: pd.DataFrame, metric_name: str = "", cmap: str = "Reds"
) -> None:
    """Plot a heatmap of first flagged (e.g. toxic) responder rounds per model.

    Args:
        heatmap_data (DataFrame): Pivot table [model x round].
        metric_name (str): Optional title suffix.
        cmap (str): Matplotlib colormap.
    """
    fig, ax = plt.subplots(figsize=(12, len(heatmap_data) * 0.6))
    cax = ax.imshow(heatmap_data.values, cmap=cmap, aspect="auto")

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)

    ax.set_xlabel("First Flagged Round")
    ax.set_ylabel("Model Name")
    ax.set_title(f"Heatmap: First Responder Turn ≥ Threshold ({metric_name})")

    # Add text labels
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iat[i, j]
            if val > 0:
                ax.text(
                    j, i, str(val), ha="center", va="center", fontsize=8, color="black"
                )

    fig.colorbar(cax, label="Dialogue Count")
    plt.tight_layout()
    plt.show()


def plot_initiator_metric_trends_by_responder_model(
    df: pd.DataFrame,
    metrics: list[str],
    include_baseline: bool = False,
    max_round: int = 12,
) -> None:
    """Plot average initiator metric scores per round, grouped by responder model.

    Args:
        df (pd.DataFrame): Full dataset.
        metrics (List[str]): Metric column names to plot.
        include_baseline (bool): Whether to include baseline dialogues.
        max_round (int): Number of dialogue rounds on x-axis.
    """
    initiator_df = df[df[ROLE] == INITIATOR]
    if not include_baseline:
        initiator_df = initiator_df[~initiator_df[IS_BASELINE]]

    # Map responder model to each chat_id
    model_map = (
        df[(df[ROLE] == RESPONDER) & (~df[IS_BASELINE])]
        .groupby(CHAT_ID)[MODEL]
        .first()
        .rename("responder_model")
    )
    initiator_df = initiator_df.merge(model_map, on=CHAT_ID, how="left")

    ncols = 4
    nrows = -(-len(metrics) // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5.5 * ncols, 4 * nrows), sharex=True
    )
    axes = axes.flatten()

    handles, labels = [], []

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Group: responder model × round
        grouped = (
            initiator_df.groupby(["responder_model", ROUND])[metric]
            .mean()
            .unstack(level=0)
            .sort_index()
        )

        for model in grouped.columns:
            (line,) = ax.plot(grouped.index, grouped[model], label=model)
            if i == 0:
                handles.append(line)
                labels.append(model)

        # Global average line
        global_mean = initiator_df.groupby(ROUND)[metric].mean()
        (global_line,) = ax.plot(
            global_mean.index,
            global_mean.values,
            linestyle="--",
            color="black",
            label="Global Mean",
        )
        if i == 0:
            handles.append(global_line)
            labels.append("Global Mean")

        # Y-axis scaling
        ymin = min(grouped.min().min(), global_mean.min())
        ymax = max(grouped.max().max(), global_mean.max())
        margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.05
        ax.set_ylim(max(0, ymin - margin), min(1.05, ymax + margin))

        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("Score")
        ax.set_xlim(1, max_round)
        ax.set_xticks(range(1, max_round + 1))
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Show x-axis labels on top & bottom row
        is_top = i // ncols == 0
        is_bottom = i // ncols == (nrows - 1)
        if is_top or is_bottom:
            ax.set_xlabel("Round")
            ax.tick_params(axis="x", labelbottom=True)

        for label in ax.get_xticklabels():
            label.set_ha("right")

    # Legend placement
    extra_slots = len(axes) - len(metrics)
    if extra_slots > 0:
        legend_ax = axes[-1]
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            labels=labels,
            loc="center",
            fontsize=10,
            title="Responder Models",
            markerscale=2,
            borderaxespad=2,
        )

    for j in range(len(metrics), len(axes) - 1):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_ranked_dialogue_counts_by_thresholds(
    df: pd.DataFrame,
    metrics: list[str],
    thresholds: list[float] = [0.5, 0.7, 0.9],
    include_baseline: bool = False,
    role: str = RESPONDER,
    fixed_ylim: int = 10,
) -> None:
    """Plot stacked bar charts of dialogues per model exceeding multiple thresholds, per metric.

    Args:
        df (pd.DataFrame): Dataset.
        metrics (List[str]): Metric columns to evaluate.
        thresholds (List[float]): Threshold cutoffs for stacking.
        include_baseline (bool): Whether to include baseline rows.
        role (str): Which role to analyze (default: 'responder').
        fixed_ylim (int): Fixed y-axis upper bound.
    """
    df_filtered = df if include_baseline else df[~df[IS_BASELINE]]
    df_role = df_filtered[df_filtered[ROLE] == role]
    model_names = sorted(df_role[MODEL].unique())

    ncols = 4
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Colors per threshold (descending)
    threshold_colors = ["#FFD700", "#FF8C00", "#DC143C"]  # gold, dark orange, crimson
    thresholds_sorted = sorted(thresholds)

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Count dialogues per threshold
        threshold_counts = {}
        for t in thresholds_sorted:
            flagged = df_role[df_role[metric] >= t]
            counts = flagged.groupby(MODEL)[CHAT_ID].nunique()
            threshold_counts[t] = counts

        bands = {}
        for i, t in enumerate(thresholds_sorted):
            if i + 1 < len(thresholds_sorted):
                next_t = thresholds_sorted[i + 1]
                band = threshold_counts[t].subtract(
                    threshold_counts[next_t], fill_value=0
                )
            else:
                band = threshold_counts[t]
            bands[t] = band.clip(lower=0)

        for t in thresholds_sorted:
            complete = pd.Series(0, index=model_names)
            complete.update(bands[t])
            bands[t] = complete

        # Plot stack
        bottom = pd.Series(0, index=model_names)
        for t, color in zip(thresholds_sorted, threshold_colors):
            ax.bar(model_names, bands[t], bottom=bottom, label=f"≥ {t}", color=color)
            bottom += bands[t]

        ax.set_title(f"{metric}", fontsize=10)
        ax.set_xlabel("Model")
        ax.set_ylabel("Dialogue Count")
        ax.set_ylim(0, fixed_ylim)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
        ax.grid(True, linestyle="--", linewidth=0.5, axis="x")

        if i == 0:
            ax.legend(loc="upper right")

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_ranked_message_counts_by_thresholds(
    df: pd.DataFrame,
    metrics: list[str],
    thresholds: list[float] = [0.5, 0.7, 0.9],
    include_baseline: bool = False,
    role: str = RESPONDER,
    fixed_ylim: int | None = None,
) -> None:
    """Plot a grid of stacked bar charts showing message counts per model, ranked by threshold bands.

    Args:
        df (pd.DataFrame): Input dataset.
        metrics (List[str]): Metric column names to evaluate.
        thresholds (List[float]): Thresholds to bin scores.
        include_baseline (bool): Include baseline rows.
        role (str): Role to filter for (default: 'responder').
        fixed_ylim (int or None): Set fixed y-axis max if provided.
    """
    df_filtered = df if include_baseline else df[~df[IS_BASELINE]]
    df_role = df_filtered[df_filtered[ROLE] == role]
    model_names = sorted(df_role[MODEL].unique())

    ncols = 4
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    threshold_colors = ["#FFD700", "#FF8C00", "#DC143C"]  # yellow, orange, red
    thresholds_sorted = sorted(thresholds)

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Count messages per threshold
        threshold_counts = {}
        for t in thresholds_sorted:
            count = df_role[df_role[metric] >= t].groupby(MODEL).size()
            threshold_counts[t] = count

        # Compute exclusive bands (avoid double counting)
        bands = {}
        for i, t in enumerate(thresholds_sorted):
            if i + 1 < len(thresholds_sorted):
                next_t = thresholds_sorted[i + 1]
                band = threshold_counts[t].subtract(
                    threshold_counts[next_t], fill_value=0
                )
            else:
                band = threshold_counts[t]
            bands[t] = band.clip(lower=0)

        # Normalize to all models
        for t in thresholds_sorted:
            complete = pd.Series(0, index=model_names)
            complete.update(bands[t])
            bands[t] = complete

        # Plot bars
        bottom = pd.Series(0, index=model_names)
        for t, color in zip(thresholds_sorted, threshold_colors):
            ax.bar(model_names, bands[t], bottom=bottom, label=f"≥ {t}", color=color)
            bottom += bands[t]

        ax.set_title(f"{metric}", fontsize=10)
        ax.set_xlabel("Model")
        ax.set_ylabel("Message Count")
        ax.set_ylim(0, max(bottom.max() + 1, 1))  # Always starts at 0; avoids flat axis

        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
        ax.grid(True, linestyle="--", linewidth=0.5, axis="x")

        if i == 0:
            ax.legend(loc="upper right")

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_responder_metric_by_round(
    df: pd.DataFrame,
    metric: str,
    severity_thresholds: list[float] = [0.5, 0.7, 0.9],
    include_baseline: bool = False,
) -> None:
    """Plot responder message scores over rounds, color-coded by severity bands.

    Args:
        df (pd.DataFrame): Full dataset.
        metric (str): Metric column to plot (e.g., 'Detoxify_toxicity').
        severity_thresholds (list[float]): Thresholds to categorize severity.
        include_baseline (bool): Whether to include baseline rows.
        title (str): Optional title override.
    """
    df_filtered = df[(df[ROLE] == RESPONDER)]
    if not include_baseline:
        df_filtered = df_filtered[~df_filtered[IS_BASELINE]]

    df_filtered = df_filtered.copy()
    df_filtered["color"] = "lightgray"

    # Color severity bands
    t1, t2, t3 = severity_thresholds
    df_filtered.loc[
        (df_filtered[metric] >= t1) & (df_filtered[metric] < t2), "color"
    ] = "#FFD700"  # yellow
    df_filtered.loc[
        (df_filtered[metric] >= t2) & (df_filtered[metric] < t3), "color"
    ] = "#FF8C00"  # orange
    df_filtered.loc[(df_filtered[metric] >= t3), "color"] = "#DC143C"  # red

    model_names = sorted(df_filtered[MODEL].unique())
    ncols = 3
    nrows = math.ceil(len(model_names) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    for i, model in enumerate(model_names):
        ax = axes[i]
        model_df = df_filtered[df_filtered[MODEL] == model]

        ax.scatter(
            model_df[ROUND],
            model_df[metric],
            c=model_df["color"],
            s=30,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
        )

        ax.set_title(model, fontsize=10)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.5, model_df[ROUND].max() + 0.5)

        xticks = range(1, int(model_df[ROUND].max()) + 1)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, linestyle="--", linewidth=0.5)

        if i // ncols == nrows - 1:
            ax.set_xlabel("Round")
        for label in ax.get_xticklabels():
            label.set_ha("right")

    for j in range(len(model_names), len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Mild ({t1}–{t2})",
            markerfacecolor="#FFD700",
            markersize=8,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Moderate ({t2}–{t3})",
            markerfacecolor="#FF8C00",
            markersize=8,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Severe (>{t3})",
            markerfacecolor="#DC143C",
            markersize=8,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Non-flagged (<{t1})",
            markerfacecolor="lightgray",
            markersize=8,
            markeredgecolor="black",
        ),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.show()
