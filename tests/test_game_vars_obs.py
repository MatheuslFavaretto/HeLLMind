"""Tests for the game-vars feature — the agent KNOWING its own health/ammo (DFP/Arnold)."""
from rl.algo import brain_prefix, gamevars_tag, policy_name


def test_gamevars_tag_on_off():
    assert gamevars_tag(True) == "_gv"
    assert gamevars_tag(False) == ""


def test_policy_name_switches_to_multiinput():
    assert policy_name(use_lstm=False, game_vars=False) == "CnnPolicy"
    assert policy_name(use_lstm=False, game_vars=True) == "MultiInputPolicy"
    assert policy_name(use_lstm=True, game_vars=False) == "CnnLstmPolicy"
    assert policy_name(use_lstm=True, game_vars=True) == "MultiInputLstmPolicy"


def test_gamevars_brain_name_distinct():
    plain = brain_prefix("campaign", 11, False)
    gv = brain_prefix("campaign", 11, False, game_vars=True)
    assert plain != gv
    assert gv == "ppo_campaign_a11_gv"


def test_gamevars_composes_with_other_flags():
    full = brain_prefix("campaign", 15, False, spatial_memory=True, game_vars=True)
    assert full == "ppo_campaign_a15_sp_gv"
