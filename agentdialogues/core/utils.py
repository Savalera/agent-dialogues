"""Agent Dialogues domain utils."""

from langchain_core.messages import AIMessage, HumanMessage

from agentdialogues.core.base import Dialogue, Roles


def convert_dialogue_to_chat_messages(
    human_role: Roles, dialogue: Dialogue
) -> list[AIMessage | HumanMessage]:
    """Map the dialogue to a series of AI and Human messages."""
    results: list[AIMessage | HumanMessage] = []

    for dialogue_item in dialogue:
        role = dialogue_item.role
        message = dialogue_item.message

        if role == human_role:
            results.append(HumanMessage(message))
        else:
            results.append(AIMessage(message))

    return results
