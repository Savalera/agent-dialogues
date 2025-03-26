#!/usr/bin/env python3
"""Simulation Command Line Runner."""

import datetime
import json
import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

from agents.simulation_agent import SimulationState
from cli import SimulationPrinter, parse_args
from domain import Roles, Simulation
from exceptions import SimulationError
from run import stream_simulation
from utils import load_simulation

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("cli")


def main() -> None:
    """Invoke simulation run."""
    console = None
    final_state: Optional[SimulationState] = None

    try:
        args = parse_args()

        sim_name = args.sim
        sim = Simulation.model_validate(load_simulation(sim_name))

        console = SimulationPrinter(
            total_steps=sim.config.rounds * 2, sim_name=sim_name
        )

        console.spinner.start()

        for chunk in stream_simulation(sim):

            message = chunk.conversation[-1].message.content
            role = chunk.conversation[-1].role
            participant_name = (
                sim.initiator.name if role == Roles.INITIATOR else sim.responder.name
            )

            console.spinner.stop()
            console.print_dialogue_message(
                role=role,
                participant_name=participant_name,
                message=str(message),
                count=len(chunk.conversation),
            )
            console.spinner.start()

            final_state = chunk

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

            log_path = log_dir / f"chat_log_{chat_id}.json"
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(chat_log, f, indent=4)

    except SimulationError as e:
        if console:
            console.spinner.stop()
        logger.exception(f"Simulation failed: {e}")
    except Exception as e:
        if console:
            console.spinner.stop()
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        if console:
            console.spinner.stop()


if __name__ == "__main__":  # pragma: no cover

    logging.basicConfig(
        level=logging.ERROR,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=False)],
    )

    main()
