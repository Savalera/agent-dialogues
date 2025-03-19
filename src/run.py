"""Command Line runner."""

import datetime
import json

from langchain_core.messages import HumanMessage

from agents import simulation as app

final_state = None

for chunk in app.stream(
    {
        "conversation": [
            {
                "role": "baby",
                "message": HumanMessage(
                    content="Daddy, please tell me about rainbows."
                ),
            }
        ]
    },
    stream_mode="values",
):
    role = chunk["conversation"][-1]["role"]
    message = chunk["conversation"][-1]["message"]
    message.type = role
    message.pretty_print()

    final_state = chunk

if final_state:
    chat_id = datetime.datetime.now().isoformat()
    chat_log = {
        "chat_id": chat_id,
        "model": "mistral-nemo",
        "conversation": [
            {"role": item["role"], "message": item["message"].content}
            for item in final_state["conversation"]
        ],
    }

    with open(f"logs/chat_log_{chat_id}.json", "w", encoding="utf-8") as f:
        json.dump(chat_log, f, indent=4)
