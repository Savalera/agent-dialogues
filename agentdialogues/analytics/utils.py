"""Analytics utils."""

from agentdialogues.analytics.config import CHAT_ID, MODEL, RESPONDER, ROLE


def get_responder_model_map(
    df, role_col=ROLE, chat_id_col=CHAT_ID, model_col=MODEL, responder_label=RESPONDER
):
    """Map each chat_id to its responder model name."""
    return (
        df[df[role_col] == responder_label]
        .groupby(chat_id_col)[model_col]
        .first()
        .rename("responder_model")
    )
