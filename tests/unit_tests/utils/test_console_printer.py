import datetime
from unittest.mock import MagicMock, patch

from rich.text import Text

from agentdialogues.core.base import Roles
from agentdialogues.utils.console_printer import SimulationPrinter


@patch("agentdialogues.utils.console_printer.Console")
def test_print_title(mock_console_cls):
    mock_console = MagicMock()
    mock_console_cls.return_value = mock_console

    printer = SimulationPrinter(
        total_steps=10,
        sim_name="test-sim",
        start_time=datetime.datetime.now(),
        batch_mode=True,
        batch_runs=5,
        output_dir="test",
        debug=False,
    )

    printer.print_title()
    assert mock_console.print.call_count >= 2


@patch("agentdialogues.utils.console_printer.Console")
def test_print_dialogue_message(mock_console_cls):
    mock_console = MagicMock()
    mock_console_cls.return_value = mock_console

    printer = SimulationPrinter(
        total_steps=6,
        sim_name="test-sim",
        start_time=datetime.datetime.now(),
        batch_mode=False,
        batch_runs=1,
        output_dir="test",
        debug=False,
    )

    printer.print_dialogue_message(
        role=Roles.INITIATOR,
        participant_name="Alice",
        message="Hi Bob.",
        meta={"toxicity": 0.1},
        count=1,
    )

    # Should call print at least for heading, message, and meta
    assert mock_console.print.call_count >= 3


@patch("agentdialogues.utils.console_printer.Console")
def test_print_batch_status(mock_console_cls):
    mock_console = MagicMock()
    mock_console_cls.return_value = mock_console

    printer = SimulationPrinter(
        total_steps=6,
        sim_name="test-sim",
        start_time=datetime.datetime.now(),
        batch_mode=True,
        batch_runs=5,
        output_dir="test",
        debug=False,
    )

    printer.print_batch_status(3)
    printed_args = [
        str(call_arg[0][0]) for call_arg in mock_console.print.call_args_list
    ]
    assert any("Running batch 3/5..." in arg for arg in printed_args)
