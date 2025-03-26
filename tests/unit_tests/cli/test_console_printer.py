from cli.console_printer import SimulationPrinter
from domain import Roles


def test_simulation_printer_initialization():
    printer = SimulationPrinter(total_steps=4, sim_name="baby-daddy")
    assert printer.total == 4
    assert printer.sim_name == "baby-daddy"
    assert hasattr(printer, "spinner")


def test_print_status_message():
    printer = SimulationPrinter(total_steps=2, sim_name="baby-dot")
    printer.print_status_message(batch_mode=True)


def test_print_dialogue_message():
    printer = SimulationPrinter(total_steps=3, sim_name="baby-bot")
    printer.print_dialogue_message(
        role=Roles.INITIATOR,
        participant_name="Alice",
        message="Hello world!",
        count=1,
    )
    printer.print_dialogue_message(
        role=Roles.RESPONDER,
        participant_name="Bob",
        message="Hi Alice!",
        count=2,
    )
