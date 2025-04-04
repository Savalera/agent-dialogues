#!/usr/bin/env python3
"""Agent Dialgoues simulation Command Line Runner."""


import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

from rich.logging import RichHandler

from agentdialogues.core.base import Roles
from agentdialogues.core.bootstrap import bootstrap_simulation
from agentdialogues.exceptions import SimulationError
from agentdialogues.utils import SimulationPrinter, parse_args

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("cli")


def main() -> None:
    """Invoke simulation run."""
    console = None
    final_state: Optional[dict[str, Any]] = None

    try:
        args = parse_args()

        simulation_module_path = args.sim
        simulation_config_path = args.config

        app, initial_state, config = bootstrap_simulation(
            simulation_module_path, simulation_config_path
        )

        console = SimulationPrinter(
            total_steps=config["runtime"]["rounds"] * 2, sim_name=config["name"]
        )

        console.spinner.start()

        for chunk in app.stream(
            initial_state,
            stream_mode="values",
        ):
            message = chunk["dialogue"][-1].message.content
            role = chunk["dialogue"][-1].role
            participant_name = (
                config["initiator"]["name"]
                if role == Roles.INITIATOR
                else config["responder"]["name"]
            )

            console.spinner.stop()
            console.print_dialogue_message(
                role=role,
                participant_name=participant_name,
                message=str(message),
                count=len(chunk["dialogue"]),
            )
            console.spinner.start()

            final_state = chunk

        if final_state:
            chat_id = datetime.datetime.now().isoformat()
            chat_log = {
                "chat_id": chat_id,
                Roles.INITIATOR: config["initiator"],
                Roles.RESPONDER: config["responder"],
                "runtime": config["runtime"],
                "dialogue": [
                    {
                        "role": item.role,
                        "name": (
                            config["initiator"]["name"]
                            if item.role == Roles.INITIATOR
                            else config["responder"]["name"]
                        ),
                        "message": item.message.content,
                    }
                    for item in final_state["dialogue"]
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
