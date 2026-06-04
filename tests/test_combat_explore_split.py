"""Tests for the combat/exploration decoupling (mode_scales) and its config wiring."""
from config import Config
from doom.campaign import mode_scales


def test_split_off_is_neutral():
    assert mode_scales(0, False, 0.25) == (1.0, 1.0)
    assert mode_scales(3, False, 0.25) == (1.0, 1.0)


def test_combat_mode_damps_exploration():
    # enemy on screen -> exploration pulls damped, combat at full strength
    explore, combat = mode_scales(2, True, 0.25)
    assert explore == 0.25
    assert combat == 1.0


def test_explore_mode_damps_combat_penalty():
    # screen clear -> exploration full, combat penalties damped
    explore, combat = mode_scales(0, True, 0.25)
    assert explore == 1.0
    assert combat == 0.25


def test_factor_passthrough():
    assert mode_scales(1, True, 0.0)[0] == 0.0   # full combat focus
    assert mode_scales(0, True, 0.5)[1] == 0.5


def test_config_exposes_split_in_reward_weights():
    w = Config().reward_weights()
    assert "combat_explore_split" in w
    assert "combat_explore_factor" in w


def test_config_split_on_by_default():
    c = Config()
    assert c.combat_explore_split is True
    assert 0.0 <= c.combat_explore_factor <= 1.0
