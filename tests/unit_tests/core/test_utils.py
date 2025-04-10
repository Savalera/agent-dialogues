from langchain_core.messages import AIMessage, HumanMessage

from agentdialogues.core.base import DialogueItem, Roles
from agentdialogues.core.utils import convert_dialogue_to_chat_messages


def test_convert_dialogue_to_chat_messages():
    dialogue = [
        DialogueItem(role=Roles.INITIATOR, message="Hello"),
        DialogueItem(role=Roles.RESPONDER, message="Hi there!"),
    ]

    result = convert_dialogue_to_chat_messages(
        human_role=Roles.INITIATOR, dialogue=dialogue
    )

    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert result[0].content == "Hello"
    assert result[1].content == "Hi there!"
