"""Simulation Agent."""

from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage
from pydantic import BaseModel, ConfigDict

from agents import dialogue_agent
from agents.dialogue_agent import State as DialogueAgentState
from domain import Roles
from exceptions import DialogueAgentError


class ConversationItem(BaseModel):
    """Conversation item."""

    role: Roles
    message: AnyMessage

    model_config = ConfigDict(arbitrary_types_allowed=True)


Conversation = List[ConversationItem]


class DialogueAgentConfig(BaseModel):
    """Dialogue agent configuration."""

    llm: Runnable
    system_prompt: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimulationState(BaseModel):
    """Simulation state."""

    conversation: Conversation
    MAX_MESSAGES: int
    initiator: DialogueAgentConfig
    responder: DialogueAgentConfig


def prepare_messages(
    human_role: Roles, conversation: List[ConversationItem]
) -> List[AIMessage | HumanMessage]:
    """Map the messages to roles before agent call."""
    results = []

    for conversation_item in conversation:
        role = conversation_item.role
        message = conversation_item.message

        if role == human_role:
            results.append(HumanMessage(message.content))
        else:
            results.append(AIMessage(message.content))

    return results


def call_dialogue_agent(
    conversation: Conversation, role: Roles, llm: Runnable, system_prompt: str
):
    """Call dialogue agent."""
    human_role = Roles.RESPONDER if role == Roles.INITIATOR else Roles.INITIATOR

    messages = prepare_messages(human_role=human_role, conversation=conversation)

    try:
        return dialogue_agent.invoke(
            {
                "messages": messages,
                "llm": llm,
                "role": role,
                "system_prompt": system_prompt,
            }
        )
    except Exception as e:
        raise DialogueAgentError(
            f"Dialogue agent invocation failed for role '{role}'."
        ) from e


def initiator_caller_node(state: SimulationState):
    """Call initiator agent."""
    response = call_dialogue_agent(
        conversation=state.conversation,
        role=Roles.INITIATOR,
        llm=state.initiator.llm,
        system_prompt=state.initiator.system_prompt,
    )

    r = DialogueAgentState(**response)
    message = r.messages[-1]

    return {
        "conversation": state.conversation
        + [ConversationItem(role=Roles.INITIATOR, message=message)]
    }


def responder_caller_node(state: SimulationState):
    """Call responder agent."""
    response = call_dialogue_agent(
        conversation=state.conversation,
        role=Roles.RESPONDER,
        llm=state.responder.llm,
        system_prompt=state.responder.system_prompt,
    )

    r = DialogueAgentState(**response)
    message = r.messages[-1]

    return {
        "conversation": state.conversation
        + [ConversationItem(role=Roles.RESPONDER, message=message)]
    }


def should_continue(state):
    """Conversation router."""
    conversation = state.conversation
    if len(conversation) >= state.MAX_MESSAGES:
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
