"""Simulation dataset builder CLI.

Combines scenario logs into one dataset based on the simulation manifest.
"""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agentdialogues.analytics.dataset_builder import (
    build_simulation_dataset_from_manifest,
)

console = Console()

# Load default directories from .env
load_dotenv()
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", "datasets"))
SIMULATIONS_DIR = Path(os.getenv("SIMULATIONS_DIR", "simulations"))


def print_summary(
    sim_id: str, output_path: Path, scenarios: list[str], individual: bool
) -> None:
    """Print dataset creation summary."""
    table = Table(title="✅ Dataset Build Summary", header_style="bold green")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Simulation ID", sim_id)
    table.add_row("Scenarios", ", ".join(scenarios))
    table.add_row("Individual Scenario Files", str(individual))
    table.add_row("Output File", str(output_path))

    console.print(table)


def main() -> None:
    """Build dataset(s)."""
    parser = argparse.ArgumentParser(
        description="Build a combined dataset for a simulation using its manifest."
    )
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        help="Simulation ID (must match folder under the simulations directory)",
    )

    args = parser.parse_args()
    simulation_id = args.sim
    manifest_path = SIMULATIONS_DIR / simulation_id / "manifest.yaml"

    if not manifest_path.exists():
        console.print(
            f"[red]❌ manifest.yaml not found for simulation: {simulation_id}[/red]"
        )
        raise SystemExit(1)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    scenario_ids = manifest.get("scenarios")
    save_individual = manifest["dataset"].get("save_scenario_datasets", False)

    if not scenario_ids:
        console.print(f"[red]No scenarios defined in {manifest_path}[/red]")
        raise SystemExit(1)

    output_path = build_simulation_dataset_from_manifest(
        simulation_id=simulation_id,
        scenario_ids=scenario_ids,
        logs_dir=LOGS_DIR,
        datasets_dir=DATASETS_DIR,
        save_scenario_datasets=save_individual,
    )

    print_summary(simulation_id, output_path, scenario_ids, save_individual)


if __name__ == "__main__":
    main()
