"""Tests for the opt-in strafe action set (ViZDoom MOVE_LEFT/MOVE_RIGHT)."""
from doom.campaign import CAMPAIGN_ACTIONS, STRAFE_ACTIONS, campaign_actions
from rl.algo import brain_prefix


def test_base_action_count_unchanged():
    assert len(campaign_actions(False)) == len(CAMPAIGN_ACTIONS) == 11


def test_strafe_appends_actions():
    acts = campaign_actions(True)
    assert len(acts) == len(CAMPAIGN_ACTIONS) + len(STRAFE_ACTIONS) == 15


def test_strafe_actions_are_appended_after_base():
    # Base indices must be preserved so a label/index never shifts meaning.
    acts = campaign_actions(True)
    assert acts[: len(CAMPAIGN_ACTIONS)] == CAMPAIGN_ACTIONS


def test_strafe_uses_only_strafe_buttons():
    for combo, label in STRAFE_ACTIONS:
        assert any(b in ("MOVE_LEFT", "MOVE_RIGHT") for b in combo)


def test_strafe_brain_name_distinct_from_base():
    # Different action count -> different `a{N}` -> brains can't cross-load.
    base = brain_prefix("campaign", 11, False, False, False)
    straf = brain_prefix("campaign", 15, False, False, False)
    assert base != straf
    assert straf == "ppo_campaign_a15"
