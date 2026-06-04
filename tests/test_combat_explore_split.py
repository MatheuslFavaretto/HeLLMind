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


def test_tracker_mode_metrics_empty_without_labels():
    from instrumentation.stats_tracker import StatsTracker
    t = StatsTracker(button_names=["ATTACK"])
    m = t._mode_metrics()
    assert m["combat_fraction"] == 0.0 and m["combat_engagement"] == 0.0


def test_tracker_mode_metrics_passive_combat():
    # Saw enemies 10% of the time but shot on only 1 of 10 combat steps -> passive.
    from instrumentation.stats_tracker import StatsTracker
    t = StatsTracker(button_names=["ATTACK"])
    t._has_mode_flag = True
    t.combat_steps, t.explore_steps = 10, 90
    t.combat_attack_steps, t.combat_hits = 1, 0.0
    m = t._mode_metrics()
    assert m["combat_fraction"] == 0.1
    assert m["combat_engagement"] == 0.1


def test_propose_uses_combat_engagement_for_passivity():
    from rl.autonomous import propose
    env = {"ENT_COEF": "0.03", "ENGAGEMENT_REWARD": "0.01", "EPISODE_TIMEOUT": "2100"}
    # Decent kills but sees enemies and won't shoot -> still flagged passive in combat.
    new, reason = propose(env, {"explored_fraction": 0.3, "death_rate": 0.1,
                                "kills_per_episode": 1.0, "timeout_rate": 0.1,
                                "combat_fraction": 0.2, "combat_engagement": 0.1})
    assert float(new["ENT_COEF"]) > 0.03
    assert "combat" in reason
