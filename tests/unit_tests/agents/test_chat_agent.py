from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agentdialogues.agents.chat_agent import (
    AgentConfig,
    AgentState,
    Runtime,
    chat_node,
    setup_node,
)


def test_setup_node_creates_runtime():
    state = AgentState(
        raw_config={"model_name": "mistral-nemo", "provider": "Ollama", "seed": 123}
    )

    result = setup_node(state)

    assert isinstance(result["config"], AgentConfig)
    assert "runtime" in result
    assert "llm" in result["runtime"]
    assert callable(result["runtime"]["llm"].invoke)


@patch("agentdialogues.agents.chat_agent.ChatOllama")
def test_chat_node_invokes_llm(mock_ollama):
    mock_llm = MagicMock(spec=Runnable)
    mock_response = AIMessage(content="Hello there!")
    mock_llm.invoke.return_value = mock_response
    mock_ollama.return_value = mock_llm

    state = AgentState(
        messages=[HumanMessage(content="Hi")],
        system_prompt="You are helpful.",
        raw_config={"model_name": "mistral-nemo", "seed": 42},
    )

    # simulate setup node call
    state.config = AgentConfig(**state.raw_config)
    state.runtime = Runtime(llm=mock_llm)

    result = chat_node(state)

    assert result["messages"][-1].content == "Hello there!"
    mock_llm.invoke.assert_called_once()
