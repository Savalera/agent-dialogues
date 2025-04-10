"""Dataset preparation cli.

Converts `json` simulation logs into a `csv` dataset.
"""

import argparse

from agentdialogues.utils import prepare_simulation_dataset


def main() -> None:
    """Entry point for cli."""
    parser = argparse.ArgumentParser(
        description="Aggregate simulation outputs from dialgoue logs."
    )
    parser.add_argument(
        "--simulation",
        type=str,
        required=True,
        help="Simulation ID (folder name under logs/)",
    )

    args = parser.parse_args()
    prepare_simulation_dataset(args.simulation)


if __name__ == "__main__":
    main()
