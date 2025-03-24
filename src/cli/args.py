"""CLI arguments handling."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a 2-agent dialogue simulation.")
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        help="Name of the simulation to load from the `sims` directory (e.g. 'baby-daddy')",
    )

    return parser.parse_args()
