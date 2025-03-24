# Agent dialogue simulations

Runs simulated dialogues between LLM agents via LangGraph.

## Why do this?

We use it for simulated conversations to check language model and agent behavior, for example:

- Does behavior change over time when chain-prompting with a certain personality?
- How well is personality preserved over time during a conversation?
- How well is role-playing preserved over time during a conversation?

## Features

Current features:

- Runs a conversation of two participants; `initiator`, `responder`.
- Configurable system prompt per participant.
- Configurable language model per participant.
- Configurable conversation length.

Known limitations:

- Local Ollama connectivity is hard coded.

Planned features:

- Classification of conversation messages for personality traits.
- Self assessment and adoption mid-conversation.

## Simulation set-up

Simulations are defined under the `sims` directory, check `sims/baby-daddy.yaml` for an example.

## How to run

1. Clone this repo.
2. Install dependencies.
3. Add your simulation configuration under `sims`.
4. Run with `run.py`.

## Output

The simulation prints the conversation output to the console, and saves a conversation log file under the `logs` directory.
