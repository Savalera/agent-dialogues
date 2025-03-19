"""Baby agent."""

from langchain_core.messages import (
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph

llm = ChatOllama(model="mistral-nemo")


def chat_node(state: MessagesState):
    """Baby chat node."""
    system = SystemMessage(
        content="""
        You are a curious baby, and you are exploring
        the things of the world and life in a conversation with your daddy
        You ask simple, but deep questions.

        You may reflect on your daddy's responses in a few short sentences, 
        but your main task is to ask one clarifying question.

        You never answer you own questions, you return questions and daddy will answer in the chat.
        """
    )

    response = llm.invoke([system] + state["messages"])
    response.type = "baby"

    return {"messages": response}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", chat_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()
