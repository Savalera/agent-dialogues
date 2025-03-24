"""Command line interface printer."""

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.text import Text

from domain import Roles

role_style = {
    Roles.INITIATOR: "bold magenta",
    Roles.RESPONDER: "bold cyan",
}


class SimulationPrinter:
    """Simulation printer."""

    def __init__(self, total_steps: int):
        """Create simulation printer."""
        self.total = total_steps
        self.console = Console()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} steps"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        self.task_id = self.progress.add_task(
            "Simulating dialogue...", total=total_steps
        )
        self.rendered = []
        self.live = Live(console=self.console, refresh_per_second=1)

    def start(self):
        """Start console UI."""
        self.progress.start()
        self.live.start()

    def update(self, role, participant_name, message):
        """Update console UI."""
        panel = Panel.fit(
            Text(message.strip(), style="white"),
            title=f"[{len(self.rendered)+1}/{self.total}] {participant_name}",
            title_align="left",
            border_style=role_style[role],
        )

        self.rendered.append(panel)
        self.live.update(Group(*self.rendered, self.progress), refresh=True)
        self.progress.update(self.task_id, advance=1)

    def stop(self):
        """Stop console UI."""
        self.progress.stop()
        self.live.stop()
        self.console.clear()
        for panel in self.rendered:
            self.console.print(panel)
