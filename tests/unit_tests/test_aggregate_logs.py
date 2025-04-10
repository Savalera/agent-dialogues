import sys
from unittest.mock import patch


@patch("agentdialogues.aggregate_logs.prepare_simulation_dataset")
@patch("agentdialogues.aggregate_logs.argparse.ArgumentParser.parse_args")
def test_dataset_cli_main(mock_parse_args, mock_prepare):
    # Simulate CLI argument parsing
    mock_parse_args.return_value.simulation = "test-sim-id"

    # Import and call main
    from agentdialogues.aggregate_logs import main

    main()

    # Ensure the prepare function was called with the simulated argument
    mock_prepare.assert_called_once_with("test-sim-id")


@patch("agentdialogues.aggregate_logs.prepare_simulation_dataset")
def test_dataset_cli_main_from_command_line(mock_prepare, monkeypatch):
    # Simulate actual command line args
    test_args = ["prog", "--simulation", "test-sim-folder"]
    monkeypatch.setattr(sys, "argv", test_args)

    from agentdialogues.aggregate_logs import main

    main()

    mock_prepare.assert_called_once_with("test-sim-folder")
