"""Command line interface printer."""

import datetime
from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from agentdialogues.core.base import Roles

BLOCK_WIDTH = 80
HEADER_BLOCK = "="

SAVALERA_LIGHT_BLUE = "#d1ecfa"
SAVALERA_LIGHT_YELLOW = "#faf7d1"
SAVALERA_PURPLE = "#cb9cfa"
SAVALERA_YELLOW = "#faea9c"

ascii_title = r"""
 __                  _
/ _\ __ ___   ____ _| | ___ _ __ __ _ 
\ \ / _` \ \ / / _` | |/ _ | '__/ _` |
_\ | (_| |\ V | (_| | |  __| | | (_| |
\__/\__,_| \_/ \__,_|_|\___|_|  \__,_|
"""

role_style = {
    Roles.INITIATOR: "bold magenta",
    Roles.RESPONDER: "bold cyan",
}


class SimulationPrinter:
    """Simulation printer."""

    def __init__(self, total_steps: int, sim_name: str):
        """Create simulation printer."""
        self.total = total_steps
        self.sim_name = sim_name
        self.console = Console()

        title = Text.from_markup(
            f"[{SAVALERA_LIGHT_YELLOW}]{ascii_title}[/{SAVALERA_LIGHT_YELLOW}]"
            + f"[{SAVALERA_LIGHT_BLUE}]Agentic Lab[/{SAVALERA_LIGHT_BLUE}] - Agent Dialogue Simulation\n[bold cyan]Running simulation [{SAVALERA_PURPLE}]`{self.sim_name}`[/{SAVALERA_PURPLE}] with [{SAVALERA_YELLOW}]{self.total}[/{SAVALERA_YELLOW}] steps."
        )
        self.console.print(title)

        status_header = self.make_heading("Job Info", SAVALERA_LIGHT_YELLOW)
        job_info = self.make_job_info()
        self.console.print(status_header)
        self.console.print(job_info)

        self.spinner = self.console.status(
            f"[{SAVALERA_LIGHT_YELLOW}]Running simulation...",
            spinner="dots",
            spinner_style=f"{SAVALERA_YELLOW}",
        )

    def make_heading(self, text: str, style: str) -> Text:
        """Print header line."""
        text = f" {text} "
        pad_len = max(0, BLOCK_WIDTH - len(text))
        left = pad_len // 2
        right = pad_len - left
        padded = HEADER_BLOCK * left + text + HEADER_BLOCK * right
        return Text(padded, style=style)

    def print_dialogue_message(
        self,
        role: Roles,
        participant_name: str,
        message: str,
        meta: Optional[dict[str, Any]],
        count: int,
    ) -> None:
        """Print dialogue message."""
        heading = self.make_heading(
            f"[{count}/{self.total}] {participant_name}", role_style[role]
        )

        self.console.print(heading)
        self.console.print(message.strip())

        if meta is not None:
            self.console.print(meta, style="dim")

    def make_job_info(self, batch_mode: bool = False) -> Table:
        """Print status message."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="bold cyan")
        table.add_column()

        table.add_row(
            "Start time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        table.add_row("Simulation", self.sim_name)
        table.add_row("Batch mode", str(batch_mode))
        table.add_row("Rounds per run", str(self.total))
        table.add_row("Total runs", str(1))
        table.add_row("Output dir", "./logs/")
        table.add_row("Debug", "False")

        return table
