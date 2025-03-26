"""Command line interface printer."""

import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from domain import Roles

ascii_title = r"""

 $$$$$$\   $$$$$$\  $$\    $$\  $$$$$$\  $$\       $$$$$$$$\ $$$$$$$\   $$$$$$\  
$$  __$$\ $$  __$$\ $$ |   $$ |$$  __$$\ $$ |      $$  _____|$$  __$$\ $$  __$$\ 
$$ /  \__|$$ /  $$ |$$ |   $$ |$$ /  $$ |$$ |      $$ |      $$ |  $$ |$$ /  $$ |
\$$$$$$\  $$$$$$$$ |\$$\  $$  |$$$$$$$$ |$$ |      $$$$$\    $$$$$$$  |$$$$$$$$ |
 \____$$\ $$  __$$ | \$$\$$  / $$  __$$ |$$ |      $$  __|   $$  __$$< $$  __$$ |
$$\   $$ |$$ |  $$ |  \$$$  /  $$ |  $$ |$$ |      $$ |      $$ |  $$ |$$ |  $$ |
\$$$$$$  |$$ |  $$ |   \$  /   $$ |  $$ |$$$$$$$$\ $$$$$$$$\ $$ |  $$ |$$ |  $$ |
 \______/ \__|  \__|    \_/    \__|  \__|\________|\________|\__|  \__|\__|  \__|

"""

role_style = {
    Roles.INITIATOR: "bold magenta",
    Roles.RESPONDER: "bold cyan",
}

SAVALERA_LIGHT_BLUE = "#d1ecfa"
SAVALERA_LIGHT_YELLOW = "#faf7d1"
SAVALERA_PURPLE = "#cb9cfa"
SAVALERA_YELLOW = "#faea9c"


class SimulationPrinter:
    """Simulation printer."""

    def __init__(self, total_steps: int, sim_name: str):
        """Create simulation printer."""
        self.total = total_steps
        self.sim_name = sim_name
        self.console = Console()

        title = Panel(
            Text.from_markup(
                f"[{SAVALERA_LIGHT_YELLOW}]{ascii_title}[/{SAVALERA_LIGHT_YELLOW}]"
                + f"\n[{SAVALERA_LIGHT_BLUE}]Agentic Lab[/{SAVALERA_LIGHT_BLUE}]\n\nAgent Dialogue Simulation\n\n[bold cyan]Running simulation [{SAVALERA_PURPLE}]`{self.sim_name}`[/{SAVALERA_PURPLE}] with [{SAVALERA_YELLOW}]{self.total}[/{SAVALERA_YELLOW}] steps.",
                justify="center",
            ),
            border_style="blue",
            expand=True,
            padding=(1, 4),
        )
        self.console.print(title)

        self.print_status_message()

        self.spinner = self.console.status(
            f"[{SAVALERA_LIGHT_YELLOW}]Running simulation...",
            spinner="dots",
            spinner_style=f"{SAVALERA_YELLOW}",
        )

    def print_dialogue_message(
        self, role: Roles, participant_name: str, message: str, count: int
    ) -> None:
        """Print dialogue message."""
        panel = Panel.fit(
            Text(message.strip(), style="white"),
            title=f"\\[{count}/{self.total}] {participant_name}",
            title_align="left",
            border_style=role_style[role],
        )
        self.console.print(panel)

    def print_status_message(self, batch_mode: bool = False) -> None:
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

        panel = Panel.fit(
            table,
            title="Job Info",
            title_align="left",
            border_style=f"bold {SAVALERA_LIGHT_YELLOW}",
        )

        self.console.print(panel)
