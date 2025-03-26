"""CLI arguments handling."""

import argparse

from exceptions import ArgumentParsingError


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a 2-agent dialogue simulation.")
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        help="Name of the simulation to load from the `sims` directory (e.g. 'baby-daddy')",
    )

    try:
        return parser.parse_args()
    except SystemExit as e:
        raise ArgumentParsingError("Failed to parse command-line arguments.") from e
