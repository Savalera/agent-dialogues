from unittest.mock import patch

from agentdialogues.agents.detoxify_agent import (
    AgentConfig,
    AgentState,
    Evaluation,
    ToxicityScore,
    detoxify_node,
    setup_node,
)


def test_setup_node_creates_config():
    state = AgentState(
        message="test message", raw_config={"model": "original", "device": "cpu"}
    )
    result = setup_node(state)
    config = result["config"]

    assert isinstance(config, AgentConfig)
    assert config.model == "original"
    assert config.device == "cpu"


@patch("agentdialogues.agents.detoxify_agent.Detoxify")
def test_detoxify_node_returns_evaluation(mock_detoxify):
    # Mock output of Detoxify.predict
    mock_instance = mock_detoxify.return_value
    mock_instance.predict.return_value = {
        "toxicity": 0.1,
        "severe_toxicity": 0.01,
        "obscene": 0.05,
        "threat": 0.0,
        "insult": 0.03,
        "identity_attack": 0.0,
        "sexual_explicit": 0.02,
    }

    state = AgentState(
        message="You're the worst!",
        raw_config={"model": "original", "device": "cpu"},
        config=AgentConfig(model="original", device="cpu"),
    )

    result = detoxify_node(state)
    evaluation = result["evaluation"]

    assert isinstance(evaluation, Evaluation)
    assert evaluation.evaluator == "Detoxify"
    assert isinstance(evaluation.score, ToxicityScore)
    assert evaluation.score.toxicity == 0.1
    mock_instance.predict.assert_called_once_with("You're the worst!")
