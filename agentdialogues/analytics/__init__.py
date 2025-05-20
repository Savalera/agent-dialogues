"""Agent Dialogues analytics."""

from .analysis import (
    analyze_first_flagged_response_round,
    analyze_initiator_intensity_vs_responder_behavior,
    compute_avg_flagged_responses_per_dialogue,
    compute_metric_profiles_by_role,
    compute_metric_summary_by_role,
    count_dialogues_with_responder_exceeding,
    describe_simulation_dataset,
    summarize_experiment_split,
    summarize_initiator_metric_by_responder,
)
from .config import DEFAULT_METRICS
from .dataset_builder import build_simulation_dataset_from_manifest
from .plots import (
    plot_dialogue_counts_above_threshold,
    plot_first_flagged_round_heatmap,
    plot_initiator_metric_trends_by_responder_model,
    plot_ranked_dialogue_counts_by_thresholds,
    plot_ranked_message_counts_by_thresholds,
)
from .preprocessing import load_dialogue_dataset

__all__ = [
    "analyze_first_flagged_response_round",
    "analyze_initiator_intensity_vs_responder_behavior",
    "compute_avg_flagged_responses_per_dialogue",
    "compute_metric_profiles_by_role",
    "compute_metric_summary_by_role",
    "count_dialogues_with_responder_exceeding",
    "describe_simulation_dataset",
    "summarize_experiment_split",
    "summarize_initiator_metric_by_responder",
    "DEFAULT_METRICS",
    "build_simulation_dataset_from_manifest",
    "plot_dialogue_counts_above_threshold",
    "plot_first_flagged_round_heatmap",
    "plot_initiator_metric_trends_by_responder_model",
    "plot_ranked_dialogue_counts_by_thresholds",
    "plot_ranked_message_counts_by_thresholds",
    "load_dialogue_dataset",
]
