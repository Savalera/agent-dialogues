# Agent dialogue simulations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15082312.svg)](https://doi.org/10.5281/zenodo.15082312)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Runs simulated dialogues between LLM agents via LangGraph.

## Why do this?

We use it for simulated conversations to check language model and agent behavior, for example:

- Does behavior change over time when chain-prompting with a certain personality?
- How well is personality preserved over time during a conversation?
- How well is role-playing preserved over time during a conversation?

## Features

### Current features:

- Runs a conversation of two participants; `initiator`, and `responder`.
- Simulation definition in `yaml` file.
  - Configurable system prompt per participant.
  - Configurable language model per participant.
  - Configurable conversation length.
- Command line interface.
- Conversation data collection via log file (`json`).

### Known limitations:

- This is an MVP, there is a lot to be added and generalized.
- Local Ollama connectivity is hard-coded.

### Planned features:

- Batch mode.
- Classification of conversation messages for personality traits.
- Self assessment and adoption mid-conversation.

## Simulation setup

Simulations are defined under the `sims` directory, check `sims/baby-daddy.yaml` for an example.

## How to run

### Run with `python3` + `pip`

```bash
# 1. Clone this repo
git clone https://github.com/savalera/agent-dialogues.git
cd agent-dialogues

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies from `requirements.txt`
pip install -r requirements.txt

# 4. Add your simulation configuration
#    Place your config in the `sims/` directory.
#    See `sims/baby-daddy.yaml` for an example.

# 5. Run the simulation
python3 src/sim-cli.py --sim baby-daddy
```

## Run with `uv`

```bash
# 1. Clone this repo
git clone https://github.com/savalera/agent-dialogues.git
cd agent-dialogues

# 2. Create a virtual environment
uv venv .venv
source .venv/bin/activate

# 3. Install dependencies from `pyproject.toml`
uv sync

# 4. Add your simulation configuration
#    Place your config in the `sims/` directory.
#    See `sims/baby-daddy.yaml` for an example.

# 5. Run the simulation
uv run src/sim-cli.py --sim baby-daddy
```

## Output

The simulation prints the conversation output to the console, and saves a conversation log file under the `logs` directory.

## Citation

If you use this project, please cite it as below:

```bibtex
@software{Takacs_Agent_Dialogues_Multi-Agent_2025,
author = {Takács, Márk},
doi = {10.5281/zenodo.15082311},
month = mar,
title = {{Agent Dialogues: Multi-Agent Simulation Framework for AI Behavior Research}},
url = {https://github.com/savalera/agent-dialogues},
version = {0.1.0-alpha.1},
year = {2025}
}
```
