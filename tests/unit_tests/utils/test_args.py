import sys

import pytest

from agentdialogues.exceptions import ArgumentParsingError
from agentdialogues.utils.args import parse_args


def test_parse_args_required(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["prog", "--sim", "bap-cla-tox", "--config", "baby-daddy.yaml"]
    )
    args = parse_args()
    assert args.sim == "bap-cla-tox"
    assert args.config == "baby-daddy.yaml"
    assert args.batch == 1
    assert args.seed is None


def test_parse_args_with_optional(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--sim",
            "bap-cla-tox",
            "--config",
            "baby-daddy.yaml",
            "--batch",
            "5",
            "--seed",
            "42",
        ],
    )
    args = parse_args()
    assert args.batch == 5
    assert args.seed == 42


def test_parse_args_missing_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--sim", "only-sim"])
    with pytest.raises(ArgumentParsingError):
        parse_args()
