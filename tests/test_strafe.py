"""Tests for the opt-in strafe action set (ViZDoom MOVE_LEFT/MOVE_RIGHT)."""
from doom.campaign import CAMPAIGN_ACTIONS, STRAFE_ACTIONS, campaign_actions
from rl.algo import brain_prefix


def test_base_action_count_unchanged():
    assert len(campaign_actions(False)) == len(CAMPAIGN_ACTIONS) == 11


def test_strafe_appends_actions():
    acts = campaign_actions(True)
    # 11 base + 8 advanced-movement (4 strafe + 4 combat-survival: dodge/retreat) = 19.
    assert len(acts) == len(CAMPAIGN_ACTIONS) + len(STRAFE_ACTIONS) == 19


def test_strafe_actions_are_appended_after_base():
    # Base indices must be preserved so a label/index never shifts meaning.
    acts = campaign_actions(True)
    assert acts[: len(CAMPAIGN_ACTIONS)] == CAMPAIGN_ACTIONS


def test_strafe_uses_advanced_movement():
    # The set is sideways (strafe) OR backward movement — the dodging/retreat the turn-only
    # base set can't express. Every combo must involve one of those movement buttons.
    for combo, label in STRAFE_ACTIONS:
        assert any(b in ("MOVE_LEFT", "MOVE_RIGHT", "MOVE_BACKWARD") for b in combo), label


def test_combat_survival_combos_keep_firing():
    # The dodge/retreat-while-firing combos must press ATTACK (so retreat stays ENGAGED — not
    # the old passive back-and-spray). BACK alone is the only no-attack retreat.
    labels = {label: combo for combo, label in STRAFE_ACTIONS}
    for lab in ("SL+ATK", "SR+ATK", "BACK+ATK"):
        assert "ATTACK" in labels[lab]


def test_strafe_brain_name_distinct_from_base():
    # Different action count -> different `a{N}` -> brains can't cross-load.
    base = brain_prefix("campaign", 11, False, False, False)
    straf = brain_prefix("campaign", 15, False, False, False)
    assert base != straf
    assert straf == "ppo_campaign_a15"
