from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from agents.dialogue_agent import State, chat_node
from exceptions import LLMInvocationError


def test_chat_node_success():
    # Mock the LLM's invoke method
    mock_llm = MagicMock(spec=Runnable)
    mock_llm.invoke.return_value = AIMessage(content="Hello from mocked LLM")

    # Setup initial state
    state = State(
        messages=[HumanMessage(content="Hello agent!")],
        llm=mock_llm,
        system_prompt="You are a helpful assistant.",
    )

    # Call chat_node
    result = chat_node(state)

    # Verify results
    mock_llm.invoke.assert_called_once_with(
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello agent!"),
        ]
    )

    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Hello from mocked LLM"


def test_chat_node_llm_failure():
    mock_llm = MagicMock(spec=Runnable)
    mock_llm.invoke.side_effect = Exception("LLM failure")

    state = State(
        messages=[HumanMessage(content="Hello agent!")],
        llm=mock_llm,
        system_prompt="You are a helpful assistant.",
    )

    # Ensure proper exception is raised
    with pytest.raises(
        LLMInvocationError, match="LLM invocation failed in dialogue agent."
    ):
        chat_node(state)

    mock_llm.invoke.assert_called_once()
