"""Analytics utils."""

import pandas as pd
from pandas import DataFrame, Series

from agentdialogues.analytics.config import CHAT_ID, MODEL, RESPONDER, ROLE


def get_responder_model_map(
    df: DataFrame,
    role_col: str = ROLE,
    chat_id_col: str = CHAT_ID,
    model_col: str = MODEL,
    responder_label: str = RESPONDER,
) -> Series[str]:
    """Map each chat_id to its responder model name."""
    return (
        df[df[role_col] == responder_label]
        .groupby(chat_id_col)[model_col]
        .first()
        .rename("responder_model")
    )
