"""Algorithm selection: PPO (feed-forward) vs RecurrentPPO (LSTM)."""
import pytest

from rl.algo import algo_class, describe, policy_name, policy_tag


def test_feedforward_selection():
    from stable_baselines3 import PPO
    assert algo_class(False) is PPO
    assert policy_name(False) == "CnnPolicy"
    assert policy_tag(False) == ""              # no tag -> existing checkpoints unaffected
    assert describe(False) == ("PPO", "CnnPolicy")


def test_lstm_selection():
    rec = pytest.importorskip("sb3_contrib")
    assert algo_class(True) is rec.RecurrentPPO
    assert policy_name(True) == "CnnLstmPolicy"
    assert policy_tag(True) == "_lstm"          # tag stops cross-loading the two families
    assert describe(True) == ("RecurrentPPO", "CnnLstmPolicy")


def test_checkpoint_name_tagging_keeps_families_apart():
    # The training name prefix must differ so resume never mixes the two brain formats.
    ff = f"ppo_campaign_a8{policy_tag(False)}"
    lstm = f"ppo_campaign_a8{policy_tag(True)}"
    assert ff == "ppo_campaign_a8"
    assert lstm == "ppo_campaign_a8_lstm"
    assert ff != lstm
