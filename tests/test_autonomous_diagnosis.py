"""The autonomous loop's rich-metric diagnoses (rl.autonomous.propose)."""
from rl.autonomous import propose


def test_spray_rule_cuts_exploration_and_sharpens_aim():
    # The exact situation we diagnosed by hand: sprays + reward is explore-dominated.
    m = {"wasted_shot_rate": 0.72, "reward_breakdown": {"explore": 0.82, "combat": 0.07},
         "aim_offset": 0.93, "explored_fraction": 0.09, "death_rate": 0.3}
    env = {"COVERAGE_REWARD": "2.0", "RND_SCALE": "0.5", "ENGAGEMENT_REWARD": "0.1",
           "MISS_PENALTY": "0.01", "FRONTIER_REWARD": "0.08"}
    new, why = propose(env, m)
    assert "spray" in why.lower()
    assert float(new["COVERAGE_REWARD"]) < 2.0           # exploration cut
    assert float(new["ENGAGEMENT_REWARD"]) > 0.1         # aim reward raised
    assert float(new["MISS_PENALTY"]) > 0.01             # trigger discipline


def test_explore_rule_skipped_when_reward_already_explore_dominant():
    # Low explored BUT reward is already ~all exploration → must NOT pour in more (the trap);
    # falls through to the spray rule instead.
    m = {"explored_fraction": 0.05, "reward_breakdown": {"explore": 0.85},
         "wasted_shot_rate": 0.6, "aim_offset": 0.9}
    env = {"COVERAGE_REWARD": "2.0", "RND_SCALE": "0.5", "ENGAGEMENT_REWARD": "0.1",
           "MISS_PENALTY": "0.01", "FRONTIER_REWARD": "0.08"}
    new, why = propose(env, m)
    assert float(new["COVERAGE_REWARD"]) < 2.0           # cut, not raised


def test_low_explore_still_raises_when_not_reward_dominant():
    # Genuinely stuck (low explored, reward NOT explore-dominated, not spraying) → raise explore.
    m = {"explored_fraction": 0.05, "reward_breakdown": {"explore": 0.2},
         "wasted_shot_rate": 0.1, "timeout_rate": 0.5}
    env = {"COVERAGE_REWARD": "1.0", "FRONTIER_REWARD": "0.05", "RND_SCALE": "0.3"}
    new, why = propose(env, m)
    assert float(new["COVERAGE_REWARD"]) > 1.0           # raised to break out (room under 1.5 cap)
