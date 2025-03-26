from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agents.simulation_agent import (
    ConversationItem,
    DialogueAgentConfig,
    SimulationState,
    call_dialogue_agent,
    initiator_caller_node,
    prepare_messages,
    responder_caller_node,
    should_continue,
)
from domain import Roles
from exceptions import DialogueAgentError


@pytest.fixture
def simulation_state():
    llm_mock = MagicMock(spec=Runnable)

    return SimulationState(
        conversation=[
            ConversationItem(
                role=Roles.RESPONDER, message=HumanMessage(content="Hello initiator!")
            )
        ],
        MAX_MESSAGES=10,
        initiator=DialogueAgentConfig(llm=llm_mock, system_prompt="Initiator prompt."),
        responder=DialogueAgentConfig(llm=llm_mock, system_prompt="Responder prompt."),
    )


def test_prepare_messages_initiator_view():
    conversation = [
        ConversationItem(role=Roles.INITIATOR, message=HumanMessage(content="Hello")),
        ConversationItem(role=Roles.RESPONDER, message=AIMessage(content="Welcome")),
    ]

    messages = prepare_messages(human_role=Roles.INITIATOR, conversation=conversation)

    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello"
    assert messages[1].content == "Welcome"


def test_prepare_messages_responder_view():
    conversation = [
        ConversationItem(role=Roles.INITIATOR, message=HumanMessage(content="Hello")),
        ConversationItem(role=Roles.RESPONDER, message=AIMessage(content="Welcome")),
    ]

    messages = prepare_messages(human_role=Roles.RESPONDER, conversation=conversation)

    assert isinstance(messages[0], AIMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[0].content == "Hello"
    assert messages[1].content == "Welcome"


@patch("agents.simulation_agent.dialogue_agent.invoke")
def test_call_dialogue_agent_success(mock_invoke):
    # Prepare mock
    mock_invoke.return_value = {"messages": [AIMessage(content="Mocked response")]}

    conversation = [
        ConversationItem(role=Roles.RESPONDER, message=HumanMessage(content="Hi"))
    ]

    response = call_dialogue_agent(
        conversation=conversation,
        role=Roles.INITIATOR,
        llm=MagicMock(spec=Runnable),
        system_prompt="Test system prompt",
    )

    assert "messages" in response
    assert isinstance(response["messages"][0], AIMessage)
    assert response["messages"][0].content == "Mocked response"

    # Ensure the mock was called with expected keys
    args, kwargs = mock_invoke.call_args
    assert isinstance(args[0], dict)
    assert "messages" in args[0]
    assert "llm" in args[0]
    assert "role" in args[0]
    assert "system_prompt" in args[0]


@patch("agents.simulation_agent.dialogue_agent.invoke")
def test_initiator_caller_node_success(mock_invoke, simulation_state):
    llm_mock = MagicMock(spec=Runnable)

    mock_invoke.return_value = {
        "llm": llm_mock,
        "messages": [AIMessage(content="Mock response from initiator")],
        "system_prompt": "Initiator prompt.",
    }

    result = initiator_caller_node(simulation_state)
    message = result["conversation"][-1].message

    assert result["conversation"][-1].role == Roles.INITIATOR
    assert isinstance(message, AIMessage)
    assert message.content == "Mock response from initiator"


@patch("agents.simulation_agent.dialogue_agent.invoke")
def test_responder_caller_node_success(mock_invoke, simulation_state):
    llm_mock = MagicMock(spec=Runnable)

    mock_invoke.return_value = {
        "llm": llm_mock,
        "messages": [AIMessage(content="Mock response from responder")],
        "system_prompt": "Responder prompt.",
    }

    result = responder_caller_node(simulation_state)
    message = result["conversation"][-1].message

    assert result["conversation"][-1].role == Roles.RESPONDER
    assert isinstance(message, AIMessage)
    assert message.content == "Mock response from responder"


@patch("agents.simulation_agent.dialogue_agent.invoke")
def test_initiator_caller_node_exception(mock_invoke, simulation_state):
    mock_invoke.side_effect = Exception("LLM failed")

    with pytest.raises(
        DialogueAgentError,
        match="Dialogue agent invocation failed for role 'Roles.INITIATOR'",
    ):
        initiator_caller_node(simulation_state)


@patch("agents.simulation_agent.dialogue_agent.invoke")
def test_responder_caller_node_exception(mock_invoke, simulation_state):
    mock_invoke.side_effect = Exception("LLM failed")

    with pytest.raises(
        DialogueAgentError,
        match="Dialogue agent invocation failed for role 'Roles.RESPONDER'",
    ):
        responder_caller_node(simulation_state)


def test_should_continue_returns_continue():
    state = SimulationState(
        conversation=[
            ConversationItem(role=Roles.INITIATOR, message=AIMessage(content="hello"))
        ],
        MAX_MESSAGES=3,
        initiator=DialogueAgentConfig(llm=MagicMock(spec=Runnable), system_prompt="a"),
        responder=DialogueAgentConfig(llm=MagicMock(spec=Runnable), system_prompt="b"),
    )
    assert should_continue(state) == "continue"


def test_should_continue_returns_end():
    state = SimulationState(
        conversation=[
            ConversationItem(role=Roles.INITIATOR, message=AIMessage(content="hi")),
            ConversationItem(role=Roles.RESPONDER, message=AIMessage(content="reply")),
            ConversationItem(role=Roles.INITIATOR, message=AIMessage(content="again")),
        ],
        MAX_MESSAGES=3,
        initiator=DialogueAgentConfig(llm=MagicMock(spec=Runnable), system_prompt="a"),
        responder=DialogueAgentConfig(llm=MagicMock(spec=Runnable), system_prompt="b"),
    )
    assert should_continue(state) == "end"
