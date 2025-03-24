#!/usr/bin/env python3
"""Simulation Command Line Runner."""

import datetime
import json
from typing import Optional

from agents.simulation_agent import SimulationState
from cli import SimulationPrinter, parse_args
from domain import Roles, Simulation
from run import stream_simulation
from utils import load_simulation


def main():
    """Invoke simulation run ."""
    args = parse_args()

    sim_name = args.sim

    sim = Simulation.model_validate(load_simulation(sim_name))

    final_state: Optional[SimulationState] = None

    printer = SimulationPrinter(total_steps=sim.config.rounds * 2)
    printer.start()

    for chunk in stream_simulation(sim):
        message = chunk.conversation[-1].message.content
        role = chunk.conversation[-1].role
        participant_name = (
            sim.initiator.name if role == Roles.INITIATOR else sim.responder.name
        )

        printer.update(role=role, participant_name=participant_name, message=message)

        final_state = chunk

    printer.stop()

    if final_state:
        chat_id = datetime.datetime.now().isoformat()
        chat_log = {
            "chat_id": chat_id,
            Roles.INITIATOR: sim.initiator.model_dump(),
            Roles.RESPONDER: sim.responder.model_dump(),
            "conversation": [
                {
                    "role": item.role,
                    "name": (
                        sim.initiator.name
                        if item.role == Roles.INITIATOR
                        else sim.responder.name
                    ),
                    "message": item.message.content,
                }
                for item in final_state.conversation
            ],
        }

        with open(f"logs/chat_log_{chat_id}.json", "w", encoding="utf-8") as f:
            json.dump(chat_log, f, indent=4)


if __name__ == "__main__":
    main()
