"""Simulation Agent."""

from typing import List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import baby, daddy

MAX_MESSAGES = 10


class ConversationItem(TypedDict):
    """Conversation item."""

    role: Literal["baby", "daddy"]
    message: AIMessage | HumanMessage


class SimulationState(TypedDict):
    """Simulation state."""

    conversation: List[ConversationItem]


def prepare_messages(
    human_role: Literal["baby", "daddy"], conversation: List[ConversationItem]
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


def baby_caller_node(state: SimulationState):
    """Baby caller node."""
    messages = prepare_messages(human_role="daddy", conversation=state["conversation"])
    response = baby.invoke({"messages": messages})

    return {
        "conversation": state["conversation"]
        + [{"role": "baby", "message": response["messages"][-1]}]
    }


def daddy_caller_node(state: SimulationState):
    """Daddy caller node."""
    messages = prepare_messages(human_role="baby", conversation=state["conversation"])
    response = daddy.invoke({"messages": messages})

    return {
        "conversation": state["conversation"]
        + [{"role": "daddy", "message": response["messages"][-1]}]
    }


def should_continue(state):
    """Conversation router."""
    conversation = state["conversation"]
    if len(conversation) >= MAX_MESSAGES:
        return "end"
    else:
        return "continue"


workflow = StateGraph(SimulationState)
workflow.add_node("baby", baby_caller_node)
workflow.add_node("daddy", daddy_caller_node)

workflow.add_edge(START, "daddy")
workflow.add_conditional_edges(
    "daddy",
    should_continue,
    {
        "end": END,
        "continue": "baby",
    },
)
workflow.add_edge("baby", "daddy")

graph = workflow.compile()
