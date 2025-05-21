# Agent dialogue simulations

[![CI](https://github.com/Savalera/agent-dialogues/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/Savalera/agent-dialogues/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/Savalera/agent-dialogues/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/Savalera/agent-dialogues/actions/workflows/integration-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15082312.svg)](https://doi.org/10.5281/zenodo.15082312)

**Agent Dialogues** is a framework for running multi-turn simulations between LLM-based agents. It's designed to be extensible by researchers and developers who want to define, run, and analyze dialogue-based interactions.

## Why do this?

We use Agent Dialogues for simulated conversations to check language model and agent behavior, for example:

- Does behavior change over time when chain-prompting with a certain personality?
- How well is personality preserved over time during a conversation?
- How well is role-play preserved over time during a conversation?

## Features

### Current features

- Runs a conversation of two participants; `initiator`, and `responder`.
- Simulation scenario definition in `yaml` file.
  - Configurable system prompt per participant.
  - Configurable language model per participant.
  - Configurable initial messages for both participants.
  - Configurable conversation length (number of rounds).
- Command line interface.
- Batch mode.
- Chat agent with [Ollama](https://github.com/ollama/ollama) support.
- Toxicity classifier agent with [Detoxify](https://github.com/unitaryai/detoxify).
- Conversation data collection via log file (in `json`).
- Log converter to `csv` dataset.
- Basic data analytics support functions to be used in Notebooks.

### Known limitations

- This is an MVP, there is a lot to be added.
- Only local Ollama invocation implemented for chat agent.

### Planned features

- Huggingface inference support.
- Big 5 personality traits evaluation.
- Sarcasm classifier.
- Self-assessment mid-conversation.
- Self-adoption during conversation.
- Improved data analytics and reporting with detailed documentation.

## Structure

The repository is structured as a Python module (`agentdialogues/`) which can be used directly for writing your own simulations.

You can:

- Use the built-in simulation runner (`sim_cli.py`).
- Write custom simulations in the `simulations/` folder.
- Import the `agentdialogues` module in your own Python project.

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/savalera/agent-dialogues.git
cd agent-dialogues

# 2. Create a virtual environment (choose one)

# Option A: Use uv (recommended for speed)
uv venv
uv pip install -r requirements.txt

# Option B: Use pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Create your own simulation
#    Place your simulation in the `simulations/` directory.
#    See `simulations/bap_chat/` or `simulations/bap_cla_tox/` for examples.

# 4. Run the simulation

# Option A: Using uv
uv run -m agentdialogues.sim_cli \
  --sim simulations/bap_chat/bap_chat.py \
  --config simulations/bap_chat/scenarios/baby-daddy.yaml \

# Option B: Using python
python3 -m agentdialogues.sim_cli \
  --sim simulations/bap_chat/bap_chat.py \
  --config simulations/bap_chat/scenarios/baby-daddy.yaml \
```

## Building a simulation

To create your own simulation using `agentdialogues`, you need to define a Python module that creates a LangGraph `graph` object at the top level.

Simulations live in the `simulations/` directory. You can use `simulations/bap_chat` or `simulations/bap_cla_tox` as reference examples.

### Requirements for a simulation module

Your simulation module must define the following:

#### 1. `graph`: a compiled LangGraph workflow

This is the object that `agentdialogues` invokes when running the simulation. Use `StateGraph(...)` to build your workflow, then call `.compile()` and assign the result to `graph`.

#### 2. Config schema

Define a Pydantic model to validate the YAML scenario config. You can:

- Use `agentdialogues.DialogueSimulationConfig` (recommended), or
- Create your own Pydantic schema

This config is passed to your simulation in `state.raw_config: dict[str, Any]`. You are responsible for validating and transforming it in the first node (typically `setup_node`).

#### 3. Simulation state

Define a `SimulationState` class using Pydantic. This defines the shape of the state passed between nodes.

Best practices:

- Include a `dialogue: Dialogue` field (a list of DialogueItems).
- Include the raw config, validated config, and runtime fields.
- Store all runtime dependencies (e.g. Runnables, encoders) in the state so LangGraph Studio can run and inspect your simulation.

State example:

```python
class SimulationState(BaseModel):
    dialogue: Dialogue = []
    raw_config: dict[str, Any] = Field(default_factory=dict)
    config: Optional[DialogueSimulationConfig] = None
    runtime: Optional[Runtime] = None
```

#### 4. Setup node

Create a setup_node that:

- Validates and parses the `raw_config`.
- Calculates runtime constants (e.g. `MAX_MESSAGES`).
- Optionally injects other objects (e.g. model seeds, loaded tools) into state.

#### 5. Build your graph

Use LangGraph primitives to define your workflow:

- Add your nodes with `add_node(...)`.
- Route control flow with `add_edge(...)` and `add_conditional_edges(...)`.
- Start and end with the `START` and `END` symbols.

Example:

```python
workflow = StateGraph(SimulationState)
workflow.add_node("setup", setup_node)
workflow.add_node("initiator", initiator_node)
workflow.add_node("responder", responder_node)

workflow.add_edge(START, "setup")
workflow.add_edge("setup", "initiator")
workflow.add_edge("initiator", "responder")
workflow.add_conditional_edges(
    "responder",
    should_continue,
    {"continue": "initiator", "end": END}
)

graph = workflow.compile()
```

### Scenarios

Every simulation can support multiple scenarios. These are variations of the same workflow with different config parameters.

Scenarios are defined in YAML under the simulation’s `scenarios/` directory.

You run a specific scenario using the command line, for example:

```bash
uv run -m agentdialogues.sim_cli \
  --sim simulations/my_simulation/main.py \
  --config simulations/my_simulation/scenarios/variant-01.yaml
```

The `--sim` argument points to the simulation module (must expose a graph variable).
The `--config` argument provides the scenario YAML file.

### Batch runs

You can repeat the same simulation scenario multiple times using the `--batch` argument:

```batch
uv run -m agentdialogues.sim_cli \
  --sim simulations/my_simulation/main.py \
  --config simulations/my_simulation/scenarios/variant-01.yaml \
  --batch 50
```

Each run will receive a different random seed and generate a separate log file. You can retrieve the seed for each run from the logs to reproduce it later.

### Simulation examples

Agent Dialogues comes with two built-in simulation examples:

- `simulations/bap_chat` - uses chat_agent to simulate a dialogue between two participants.
- `simulations/bap_cla_tox` - uses chat agent to simulate a dialogue and applies Detoxify toxicity classification on every message.

## Running Your Simulation in LangGraph Studio

To run your simulation interactively in LangGraph Studio, you can register it in the `langgraph.json` file:

```json
{
  "bap_chat": "./simulations/bap_chat/bap_chat.py:graph"
}
```

The key (bap_chat) is the name of your simulation, and the value is the path to your module followed by :graph, referring to the compiled graph object.

Once registered, you can launch LangGraph Studio with:

```bash
uv run langgraph dev
```

For more information on setup and advanced usage, see the [LangGraph Studio documentation](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/).

## Built-in Agents

Agent Dialogues includes a set of built-in agents designed to cover common simulation use cases:

### 1. `chat_agent`

This agent is used for LLM-based conversational turns. It currently supports:

- Local Ollama.
- Planned Hugging Face support.

You can customize the model via the `model_name` and `provider` fields in the simulation config. The agent accepts messages, system prompts, and a seed for reproducibility.

### 2. `detoxify_agent`

This agent performs toxicity classification using the [Detoxify](https://github.com/unitaryai/detoxify) model locally. It requires no external API and supports GPU acceleration via PyTorch.

The [simulations/bap_cla_tox/bap_cla_tox.py](simulations/bap_cla_tox/bap_cla_tox.py) example demonstrates a dialogue simulation where each message is followed by a toxicity classification.

## Core schemas

Agent Dialogues revolves around two main data structures:

- `Dialogue` — captures the conversation.
- `DialogueSimulationConfig` — defines the setup for a full simulation.

### Dialogue format

A `Dialogue` is a list of turns between two agents. Each turn is represented as a `DialogueItem`, which includes:

```python
DialogueItem:
  role: Roles # either "initiator" or "responder"
  message: str
  meta: Optional[list[dict[str, Any]]]
```

The meta field can be used to store arbitrary structured annotations, such as evaluation results or classifier outputs.

Example:

```json
[
  {
    "role": "initiator",
    "message": "What is love?",
    "meta": [{ "toxicity": 0.01 }]
  },
  {
    "role": "responder",
    "message": "Love is a deep emotional connection.",
    "meta": [{ "toxicity": 0.01 }]
  }
]
```

### DialogueSimulationConfig

This schema defines the full setup for a simulation and is used to validate the scenario YAML file. It ensures consistency in how agents and their behavior are configured.

```python
DialogueSimulationConfig:
  id: str                      # Unique simulation ID
  name: str                    # Human-readable name for logs/UI
  seed: int                    # Global seed for reproducibility

  initiator: DialogueParticipantWithMessagesConfig
  responder: DialogueParticipantConfig

  runtime: RuntimeConfig
  evaluation: Optional[dict[str, Any]]
```

#### Agent definitions

Both participants use a similar schema to define their behavior and models:

```python
DialogueParticipantConfig:
  name: str
  role: str
  model:
    provider: "Ollama" | "HuggingFace" | "HFApi"
    model_name: str
  system_prompt: str
```

For the initiator, you can optionally provide a list of seed messages:

```python
DialogueParticipantWithMessagesConfig:
  messages: Optional[list[str]]
```

These messages are injected during the dialogue, one message per turn. This way you can seed messages in several turns into a simulation.

#### Runtime settings

The number of dialogue turns is specified in the runtime block:

```python
RuntimeConfig:
  rounds: int  # number of dialogue rounds (initiator+responder = 2x)
```

#### Evaluation settings

Optionally, you can define evaluation steps — for example, to run classifiers on messages.

```yaml
evaluation:
  detoxify:
    model: unbiased
    device: mps
```

## Data and analytics

Simulation runs are automatically logged under the `logs/` directory.

Each scenario has its own subfolder, and each individual run is saved as a separate `.json` file.

To analyze results across multiple runs, use the built-in aggregation script:

```bash
python3 -m agentdialogues.aggregate_logs.py --simulation logs/baby-daddy
```

This creates an aggregated_scores.csv file in the scenario’s log folder, containing flattened data suitable for further analysis.

The project also provides a `/notebooks` directory where you can store and run Jupyter notebooks.

Support for analytics helper functions (e.g., DataFrames, plotting) is implemented in the analytics module (currently in alpha).

## Citation

If you use this project, please cite it as below:

```bibtex
@software{Takacs_Agent_Dialogues_Multi-Agent_2025,
author = {Takács, Márk},
doi = {10.5281/zenodo.15082311},
month = mar,
title = {{Agent Dialogues: Multi-Agent Simulation Framework for AI Behavior Research}},
url = {https://github.com/savalera/agent-dialogues},
version = {0.1.0-alpha.2},
year = {2025}
}
```
