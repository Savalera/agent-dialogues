from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agents.simulation_agent import (
    ConversationItem,
    DialogueAgentConfig,
    SimulationState,
)
from domain import (
    DialogueInitiator,
    DialogueParticipant,
    Roles,
    Simulation,
    SimulationConfig,
)
from sim_cli import main as sim_cli


@pytest.fixture
def fake_simulation():
    return Simulation(
        id="baby-daddy",
        description="Test simulation",
        initiator=DialogueInitiator(
            name="Alice",
            persona="lead",
            model_name="mistral-nemo",
            system_prompt="Be clear.",
            initial_message="Hi Bob.",
        ),
        responder=DialogueParticipant(
            name="Bob",
            persona="dev",
            model_name="mistral-nemo",
            system_prompt="Be clear.",
        ),
        config=SimulationConfig(rounds=1),
    )


@patch("sim_cli.load_simulation")
@patch("sim_cli.parse_args")
@patch("sim_cli.stream_simulation")
@patch("sim_cli.SimulationPrinter")
@patch.object(Path, "open", autospec=True)
def test_cli_main_success(
    mock_path_open,
    mock_printer_class,
    mock_stream,
    mock_args,
    mock_load_sim,
    fake_simulation,
):
    # Mock CLI args
    mock_args.return_value.sim = "baby-daddy"
    mock_load_sim.return_value = fake_simulation

    # Mock message stream
    mock_state = SimulationState(
        conversation=[
            ConversationItem(
                role=Roles.INITIATOR, message=HumanMessage(content="Hi Bob.")
            ),
            ConversationItem(
                role=Roles.RESPONDER, message=AIMessage(content="Hi Alice.")
            ),
        ],
        MAX_MESSAGES=2,
        initiator=DialogueAgentConfig(
            llm=MagicMock(spec=Runnable), system_prompt="Be clear."
        ),
        responder=DialogueAgentConfig(
            llm=MagicMock(spec=Runnable), system_prompt="Be clear."
        ),
    )

    mock_stream.return_value = iter([mock_state])

    mock_printer = MagicMock()
    mock_printer_class.return_value = mock_printer
    mock_file = MagicMock()
    mock_path_open.return_value.__enter__.return_value = mock_file

    # Run the main function
    sim_cli()

    # Verify spinner start/stop
    mock_printer.spinner.start.assert_called()
    mock_printer.spinner.stop.assert_called()

    # Verify dialogue message was printed
    mock_printer.print_dialogue_message.assert_called()

    # Chat log written
    mock_path_open.assert_called_once()
    mock_file.write.assert_called()

    # ✅ Extract `self` (the Path instance) from the bound method call
    path_instance = mock_path_open.call_args_list[0][0][0]
    assert isinstance(path_instance, Path)
    assert str(path_instance).startswith("logs/chat_log_")


@patch("sim_cli.load_simulation", side_effect=RuntimeError("Sim fail"))
@patch("sim_cli.parse_args")
@patch("sim_cli.SimulationPrinter")
def test_catch_unexpected_exception(mock_printer_cls, mock_parse, mock_load):
    mock_parse.return_value.sim = "whatever"
    mock_printer = MagicMock()
    mock_printer_cls.return_value = mock_printer

    sim_cli()

    mock_printer_cls.assert_not_called()
