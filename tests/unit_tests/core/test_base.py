from agentdialogues.core.base import (
    ChatModelConfig,
    ChatProviders,
    Dialogue,
    DialogueItem,
    DialogueParticipantConfig,
    DialogueParticipantWithMessagesConfig,
    DialogueSimulationConfig,
    Roles,
    RuntimeConfig,
)


# === Enum Tests ===
def test_roles_enum_values():
    assert Roles.INITIATOR.value == "initiator"
    assert Roles.RESPONDER.value == "responder"


def test_chat_providers_enum_values():
    assert ChatProviders.OLLAMA.value == "Ollama"
    assert ChatProviders.HFAPI.value == "HFApi"
    assert ChatProviders.HUGGINGFACE.value == "Huggingface"


# === Dialogue Tests ===
def test_dialogue_type_alias():
    item1 = DialogueItem(role=Roles.INITIATOR, message="Hi")
    item2 = DialogueItem(role=Roles.RESPONDER, message="Hello")
    dialogue: Dialogue = [item1, item2]

    assert isinstance(dialogue, list)
    assert all(isinstance(item, DialogueItem) for item in dialogue)
    assert dialogue[0].role == Roles.INITIATOR
    assert dialogue[1].message == "Hello"


# === DialogueItem Tests ===
def test_dialogue_item_instantiation():
    item = DialogueItem(role=Roles.INITIATOR, message="Hello")
    assert item.role == Roles.INITIATOR
    assert item.message == "Hello"
    assert item.meta is None


def test_dialogue_item_with_meta():
    meta = [{"toxicity": 0.1}]
    item = DialogueItem(role=Roles.RESPONDER, message="Hi", meta=meta)
    assert item.meta == meta


# === Participant and Config Tests ===
def test_chat_model_config():
    config = ChatModelConfig(model_name="mistral", provider=ChatProviders.OLLAMA)
    assert config.model_name == "mistral"
    assert config.provider == ChatProviders.OLLAMA


def test_runtime_config():
    runtime = RuntimeConfig(rounds=5)
    assert runtime.rounds == 5


def test_dialogue_participant_config():
    participant = DialogueParticipantConfig(
        name="Alice",
        role="initiator",
        model=ChatModelConfig(model_name="mistral", provider=ChatProviders.OLLAMA),
        system_prompt="You are helpful.",
    )
    assert participant.name == "Alice"
    assert participant.role == "initiator"
    assert participant.model.provider == ChatProviders.OLLAMA


def test_dialogue_participant_with_messages_defaults():
    participant = DialogueParticipantWithMessagesConfig(
        name="Bob",
        role="responder",
        model=ChatModelConfig(model_name="mistral", provider=ChatProviders.OLLAMA),
        system_prompt="Be concise.",
    )
    assert isinstance(participant.messages, list)
    assert participant.messages == []


def test_dialogue_participant_with_messages_provided():
    participant = DialogueParticipantWithMessagesConfig(
        name="Bob",
        role="responder",
        model=ChatModelConfig(model_name="mistral", provider=ChatProviders.OLLAMA),
        system_prompt="Be concise.",
        messages=["Hello", "Can you help?"],
    )
    assert participant.messages == ["Hello", "Can you help?"]


def test_dialogue_simulation_config_optional_fields():
    sim = DialogueSimulationConfig(
        id="sim1",
        name="Sample Simulation",
        seed=42,
        initiator=DialogueParticipantWithMessagesConfig(
            name="Init",
            role="initiator",
            model=ChatModelConfig(model_name="mistral", provider=ChatProviders.OLLAMA),
            system_prompt="Start the convo",
            messages=None,  # Explicitly passing None
        ),
        responder=DialogueParticipantConfig(
            name="Resp",
            role="responder",
            model=ChatModelConfig(model_name="mistral", provider=ChatProviders.OLLAMA),
            system_prompt="Reply smartly",
        ),
        runtime=RuntimeConfig(rounds=2),
        evaluation=None,  # Optional field omitted
    )
    assert sim.id == "sim1"
    assert sim.initiator.messages is None or isinstance(sim.initiator.messages, list)
    assert sim.runtime.rounds == 2
    assert sim.evaluation is None
