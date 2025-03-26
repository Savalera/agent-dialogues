from unittest.mock import patch

import pytest

from cli.args import parse_args
from exceptions import ArgumentParsingError


@patch("sys.argv", ["prog", "--sim", "baby-daddy"])
def test_parse_args_success():
    args = parse_args()
    assert args.sim == "baby-daddy"


@patch("sys.argv", ["prog"])  # Missing --sim
def test_parse_args_missing_sim():
    with pytest.raises(
        ArgumentParsingError, match="Failed to parse command-line arguments."
    ):
        parse_args()
