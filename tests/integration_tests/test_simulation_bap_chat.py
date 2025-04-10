from unittest.mock import patch

from langchain_core.messages import AIMessage

from agentdialogues.core.base import Roles
from simulations.bap_chat import bap_chat


@patch("simulations.bap_chat.bap_chat.chat_agent.invoke")
def test_bap_chat_runs_offline(mock_chat_invoke):
    """Run bap_chat simulation with mocked chat agent (offline, CPU-safe)."""
    # Prepare mock response
    mock_chat_invoke.return_value = {
        "messages": [AIMessage(content="This is a mocked response.")]
    }

    raw_config = {
        "id": "test_bap_chat",
        "name": "Test Bap Chat",
        "seed": 1234,
        "runtime": {"rounds": 2},
        "initiator": {
            "name": "Student",
            "role": Roles.INITIATOR,
            "model": {"model_name": "mistral", "provider": "Ollama"},
            "system_prompt": "You are a curious student.",
            "messages": ["What is gravity?"],
        },
        "responder": {
            "name": "Teacher",
            "role": Roles.RESPONDER,
            "model": {"model_name": "mistral", "provider": "Ollama"},
            "system_prompt": "You are a wise teacher.",
        },
    }

    state = {"dialogue": [], "raw_config": raw_config}

    final_state = None
    for chunk in bap_chat.graph.stream(state, stream_mode="values"):
        final_state = chunk

    assert final_state is not None
    dialogue = final_state["dialogue"]
    assert len(dialogue) == 4  # 2 rounds = 4 messages
    for msg in dialogue:
        assert msg.role in {Roles.INITIATOR, Roles.RESPONDER}
        assert isinstance(msg.message, str)

    responder_msgs = [msg for msg in dialogue if msg.role == Roles.RESPONDER]
    for msg in responder_msgs:
        assert "mocked response" in msg.message.lower()
