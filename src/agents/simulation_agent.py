"""Simulation Agent."""

from typing import Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import Messages

from agents import dialogue_agent
from constants import Roles


class ConversationItem(TypedDict):
    """Conversation item."""

    role: Roles
    message: AIMessage | HumanMessage


class SimulationState(TypedDict):
    """Simulation state."""

    conversation: List[ConversationItem]
    MAX_MESSAGES: int
    initiator: Dict[str, Any]
    responder: Dict[str, Any]


def prepare_messages(
    human_role: Roles, conversation: List[ConversationItem]
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


def call_dialogue_agent(
    conversation: List[ConversationItem], role: Roles, llm: Any, system_prompt: str
):
    """Call dialogue agent."""
    human_role = Roles.RESPONDER if role == Roles.INITIATOR else Roles.INITIATOR

    messages = prepare_messages(human_role=human_role, conversation=conversation)

    return dialogue_agent.invoke(
        {
            "messages": messages,
            "llm": llm,
            "role": role,
            "system_prompt": system_prompt,
        }
    )


def initiator_caller_node(state: SimulationState):
    """Call initiator agent."""
    initiator = state[Roles.INITIATOR.value]
    conversation = state["conversation"]

    response = call_dialogue_agent(
        conversation=conversation,
        role=Roles.INITIATOR,
        llm=initiator["llm"],
        system_prompt=initiator["system_prompt"],
    )

    return {
        "conversation": state["conversation"]
        + [{"role": Roles.INITIATOR, "message": response["messages"][-1]}]
    }


def responder_caller_node(state: SimulationState):
    """Call responder agent."""
    responder = state[Roles.RESPONDER.value]
    conversation = state["conversation"]

    response = call_dialogue_agent(
        conversation=conversation,
        role=Roles.RESPONDER,
        llm=responder["llm"],
        system_prompt=responder["system_prompt"],
    )

    return {
        "conversation": state["conversation"]
        + [{"role": Roles.RESPONDER, "message": response["messages"][-1]}]
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
