"""Command Line runner."""

import datetime
import json

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agents import simulation_agent as app
from config import load_simulation

sim = load_simulation("baby-daddy")

initiator = sim["initiator"]
responder = sim["responder"]

initial_message = sim["initiator"]["initial_message"]

final_state = None

for chunk in app.stream(
    {
        "conversation": [
            {
                "role": "initiator",
                "message": HumanMessage(content=initial_message),
            }
        ],
        "MAX_MESSAGES": sim["config"]["rounds"] * 2,
        "initiator": {
            "llm": ChatOllama(model=initiator["model_name"]),
            **initiator,
        },
        "responder": {"llm": ChatOllama(model=responder["model_name"]), **responder},
    },
    stream_mode="values",
):

    role = (
        f"""{initiator["name"]} (Initiator)"""
        if chunk["conversation"][-1]["role"] == "initiator"
        else f"""{responder["name"]} (Responder)"""
    )

    message = chunk["conversation"][-1]["message"]
    message.type = role
    message.pretty_print()

    final_state = chunk

if final_state:
    chat_id = datetime.datetime.now().isoformat()
    chat_log = {
        "chat_id": chat_id,
        "initiator": sim["initiator"],
        "responder": sim["responder"],
        "conversation": [
            {
                "role": item["role"],
                "name": (
                    initiator["name"]
                    if item["role"] == "initiator"
                    else responder["name"]
                ),
                "message": item["message"].content,
            }
            for item in final_state["conversation"]
        ],
    }

    with open(f"logs/chat_log_{chat_id}.json", "w", encoding="utf-8") as f:
        json.dump(chat_log, f, indent=4)
