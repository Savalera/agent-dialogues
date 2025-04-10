from unittest.mock import MagicMock, mock_open, patch

from agentdialogues.core.base import Roles


@patch("agentdialogues.sim_cli.bootstrap_simulation")
@patch("agentdialogues.sim_cli.parse_args")
@patch("agentdialogues.sim_cli.SimulationPrinter")
@patch("agentdialogues.sim_cli.Path.open", new_callable=mock_open)
def test_main_single_run(mock_open, mock_printer_cls, mock_parse_args, mock_bootstrap):
    mock_args = MagicMock(sim="sim.py", config="scenario.yaml", batch=1, seed=1234)
    mock_parse_args.return_value = mock_args

    mock_graph = MagicMock()
    mock_stream = [
        {
            "dialogue": [
                MagicMock(role=Roles.INITIATOR, message="Hello", meta=None),
            ]
        }
    ]
    mock_graph.stream.return_value = iter(mock_stream)

    mock_config = {
        "id": "test_sim",
        "name": "Test Simulation",
        "runtime": {"rounds": 1},
        "initiator": {"name": "Alice"},
        "responder": {"name": "Bob"},
    }
    mock_bootstrap.return_value = (mock_graph, mock_config)

    mock_console = MagicMock()
    mock_printer_cls.return_value = mock_console
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    from agentdialogues.sim_cli import main

    main()

    mock_console.print_title.assert_called_once()
    assert mock_console.print_dialogue_message.call_count == 1
    mock_open.assert_called_once()
    mock_file.write.assert_called()


@patch("agentdialogues.sim_cli.bootstrap_simulation")
@patch("agentdialogues.sim_cli.parse_args")
@patch("agentdialogues.sim_cli.SimulationPrinter")
@patch("agentdialogues.sim_cli.Path.open", new_callable=mock_open)
def test_main_handles_simulation_error(
    mock_open, mock_printer_cls, mock_parse_args, mock_bootstrap
):
    mock_args = MagicMock(sim="sim.py", config="scenario.yaml", batch=1, seed=None)
    mock_parse_args.return_value = mock_args

    mock_bootstrap.side_effect = Exception("Boom")

    from agentdialogues.sim_cli import main

    main()

    # Printer should never be constructed due to early failure
    mock_printer_cls.assert_not_called()


@patch("agentdialogues.sim_cli.bootstrap_simulation")
@patch("agentdialogues.sim_cli.parse_args")
@patch("agentdialogues.sim_cli.SimulationPrinter")
@patch("agentdialogues.sim_cli.Path.open", new_callable=mock_open)
def test_main_batch_runs(mock_open, mock_printer_cls, mock_parse_args, mock_bootstrap):
    mock_args = MagicMock(sim="sim.py", config="scenario.yaml", batch=2, seed=None)
    mock_parse_args.return_value = mock_args

    mock_graph = MagicMock()

    # Yield new stream for each batch run
    def mock_stream_gen(*args, **kwargs):
        yield {
            "dialogue": [
                MagicMock(role=Roles.INITIATOR, message="Hi", meta=None),
            ]
        }

    mock_graph.stream.side_effect = [iter(mock_stream_gen()), iter(mock_stream_gen())]

    mock_config = {
        "id": "test_sim",
        "name": "Test Simulation",
        "runtime": {"rounds": 1},
        "initiator": {"name": "Alice"},
        "responder": {"name": "Bob"},
    }
    mock_bootstrap.return_value = (mock_graph, mock_config)

    mock_console = MagicMock()
    mock_printer_cls.return_value = mock_console
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    from agentdialogues.sim_cli import main

    main()

    assert mock_console.print_batch_status.call_count == 2
    assert mock_console.print_dialogue_message.call_count == 2
    assert mock_open.call_count == 2
