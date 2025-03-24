"""Simulation Agent."""

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import dialogue_agent


class ConversationItem(TypedDict):
    """Conversation item."""

    role: Literal["initiator", "responder"]
    message: AIMessage | HumanMessage


class SimulationState(TypedDict):
    """Simulation state."""

    conversation: List[ConversationItem]
    MAX_MESSAGES: int
    initiator: Dict[str, Any]
    responder: Dict[str, Any]


def prepare_messages(
    human_role: Literal["initiator", "responder"], conversation: List[ConversationItem]
) -> List[AIMessage | HumanMessage]:
    """Map the messages to roles before agent call."""
    results = []

    for conversation_item in conversation:
        role = conversation_item["role"]
        message = conversation_item["message"]

        if role == human_role:
            results.append(HumanMessage(message.content))
        else:
            results.append(AIMessage(message.content))

    return results


def initiator_caller_node(state: SimulationState):
    """Call initiator agent."""
    initiator = state["initiator"]
    conversation = state["conversation"]

    messages = prepare_messages(human_role="responder", conversation=conversation)
    response = dialogue_agent.invoke(
        {
            "messages": messages,
            **initiator,
        }
    )

    return {
        "conversation": state["conversation"]
        + [{"role": "initiator", "message": response["messages"][-1]}]
    }


def responder_caller_node(state: SimulationState):
    """Call responder agent."""
    responder = state["responder"]
    conversation = state["conversation"]

    messages = prepare_messages(human_role="initiator", conversation=conversation)
    response = dialogue_agent.invoke(
        {
            "messages": messages,
            **responder,
        }
    )

    return {
        "conversation": state["conversation"]
        + [{"role": "responder", "message": response["messages"][-1]}]
    }


def should_continue(state):
    """Conversation router."""
    conversation = state["conversation"]
    if len(conversation) >= state["MAX_MESSAGES"]:
        return "end"
    else:
        return "continue"


workflow = StateGraph(SimulationState)
workflow.add_node("initiator_node", initiator_caller_node)
workflow.add_node("responder_node", responder_caller_node)

workflow.add_edge(START, "responder_node")
workflow.add_conditional_edges(
    "responder_node",
    should_continue,
    {
        "end": END,
        "continue": "initiator_node",
    },
)
workflow.add_edge("initiator_node", "responder_node")

graph = workflow.compile()
