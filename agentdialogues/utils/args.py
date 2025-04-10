"""CLI arguments handling."""

import argparse

from agentdialogues.exceptions import ArgumentParsingError


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a 2-agent dialogue simulation.")
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        help="Name of the simulation implementation to load from the `src/simulations` directory (e.g. 'bap-cla-tox')",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the scenario sim file to load from the `sims` directory (e.g. 'baby-daddy')",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Run in batch mode with N repetitions. 0 means run once interactively.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility. (Only applied when not in batch mode.)",
    )

    try:
        return parser.parse_args()
    except SystemExit as e:
        raise ArgumentParsingError("Failed to parse command-line arguments.") from e
