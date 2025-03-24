"""Command line interface printer."""

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
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


class SimulationPrinter:
    """Simulation printer."""

    def __init__(self, total_steps: int, sim_name: str):
        """Create simulation printer."""
        self.total = total_steps
        self.sim_name = sim_name
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
        self.live = Live(console=self.console, refresh_per_second=1, transient=True)

    def start(self):
        """Start console UI."""
        self.progress.start()
        self.live.start()

        panel = Panel(
            Text.from_markup(
                f"[#faf7d1]{ascii_title}[/#faf7d1]"
                + f"\n[#d1ecfa]Agentic Lab[/#d1ecfa]\n\nAgent Dialogue Simulation\n\n[bold cyan]Running simulation [#cb9cfa]`{self.sim_name}`[/#cb9cfa] with [#faea9c]{self.total}[/#faea9c] steps.",
                justify="center",
            ),
            border_style="blue",
            expand=True,
            padding=(1, 4),
        )

        self.rendered.append(panel)
        self.live.update(Group(*self.rendered, self.progress), refresh=True)

    def update(self, role, participant_name, message):
        """Update console UI."""
        panel = Panel.fit(
            Text(message.strip(), style="white"),
            title=f"[{len(self.rendered)}/{self.total}] {participant_name}",
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
