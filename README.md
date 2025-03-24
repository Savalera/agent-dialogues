# Agent dialogue simulations

Runs simulated dialogues between LLM agents via LangGraph.

## Why do this?

We use it for simulated conversations to check language model and agent behavior, for example:

- Does behavior change over time when chain-prompting with a certain personality?
- How well is personality preserved over time during a conversation?
- How well is role-playing preserved over time during a conversation?

## Features

Current features:

- Runs a conversation of two participants; `initiator`, and `responder`.
- Simulation definition in `yaml` file.
  - Configurable system prompt per participant.
  - Configurable language model per participant.
  - Configurable conversation length.
- Command line interface.
- Conversation data collection via log file (`json`).

Known limitations:

- This is an MVP, there is a lot to be added and generalized.
- Local Ollama connectivity is hard-coded.

Planned features:

- Classification of conversation messages for personality traits.
- Self assessment and adoption mid-conversation.

## Simulation set-up

Simulations are defined under the `sims` directory, check `sims/baby-daddy.yaml` for an example.

## How to run

1. Clone this repo.
2. Install dependencies.
3. Add your simulation configuration in the `sims` directory.
4. Run with `src/sim-cli.py` with the `sim` argument, for example:

```sh
uv run src/sim-cli.py --sim baby-daddy
```

## Output

The simulation prints the conversation output to the console, and saves a conversation log file under the `logs` directory.
