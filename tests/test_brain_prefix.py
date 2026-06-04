"""Tests for rl.algo.brain_prefix / spatial_tag — checkpoint-name family guards.

Regression: a spatial-memory brain (2 obs channels) and a plain brain (1 channel) once
shared the name `ppo_campaign_a11` and cross-loaded → hard obs-shape crash on eval/resume.
The `_sp` tag must keep the families apart, exactly like `_lstm`.
"""
from rl.algo import automap_tag, brain_prefix, depth_tag, policy_tag, spatial_tag


def test_automap_tag_on_off():
    assert automap_tag(True) == "_am"
    assert automap_tag(False) == ""


def test_automap_brain_distinct_and_composes():
    assert brain_prefix("campaign", 11, False, automap=True) == "ppo_campaign_a11_am"
    # all obs-shape channels compose into one unambiguous family name
    full = brain_prefix("campaign", 11, True, spatial_memory=True,
                        depth_perception=True, automap=True)
    assert full == "ppo_campaign_a11_lstm_sp_dp_am"


def test_framestack_tag_default_untagged():
    from rl.algo import framestack_tag
    assert framestack_tag(4) == ""          # default stays untagged (no rename of old brains)
    assert framestack_tag(2) == "_fs2"
    assert framestack_tag(1) == "_fs1"


def test_framestack_changes_brain_name():
    # A different frame_stack changes the obs shape -> must not collide with the default.
    default = brain_prefix("campaign", 11, False, frame_stack=4)
    fs2 = brain_prefix("campaign", 11, False, frame_stack=2)
    assert default == "ppo_campaign_a11"
    assert fs2 == "ppo_campaign_a11_fs2"
    assert default != fs2


def test_spatial_tag_on_off():
    assert spatial_tag(True) == "_sp"
    assert spatial_tag(False) == ""


def test_depth_tag_on_off():
    assert depth_tag(True) == "_dp"
    assert depth_tag(False) == ""


def test_depth_brain_distinct():
    assert brain_prefix("campaign", 11, False, False, True) == "ppo_campaign_a11_dp"
    # depth + spatial compose, and never collide with either alone
    names = {
        brain_prefix("campaign", 11, False, False, False),
        brain_prefix("campaign", 11, False, True, False),
        brain_prefix("campaign", 11, False, False, True),
        brain_prefix("campaign", 11, False, True, True),
    }
    assert len(names) == 4  # all four obs-shape families are distinct


def test_policy_tag_on_off():
    assert policy_tag(True) == "_lstm"
    assert policy_tag(False) == ""


def test_plain_brain_name():
    assert brain_prefix("campaign", 11, False, False) == "ppo_campaign_a11"


def test_spatial_brain_distinct_from_plain():
    plain = brain_prefix("campaign", 11, False, False)
    spatial = brain_prefix("campaign", 11, False, True)
    assert plain != spatial
    assert spatial == "ppo_campaign_a11_sp"


def test_lstm_and_spatial_compose():
    assert brain_prefix("campaign", 11, True, True) == "ppo_campaign_a11_lstm_sp"


def test_action_count_in_name():
    assert brain_prefix("campaign", 8, False, False) == "ppo_campaign_a8"
    # Different action counts never collide.
    assert brain_prefix("campaign", 8, False, False) != brain_prefix("campaign", 11, False, False)


def test_scenario_task_name():
    assert brain_prefix("defend_the_center", 3, False, False) == "ppo_defend_the_center_a3"
