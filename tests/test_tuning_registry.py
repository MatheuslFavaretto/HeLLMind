"""Tests for rl.tuning_registry — the full tunable-parameter catalog for the LLM/auto-loop."""
from rl.tuning_registry import PARAMS, all_bounds, clamp, validate, describe_for_llm


def test_registry_covers_key_knobs():
    envs = {p["env"] for p in PARAMS}
    for k in ("KILL_REWARD", "ENGAGEMENT_REWARD", "COVERAGE_REWARD", "DEATH_PENALTY",
              "ENT_COEF", "EXIT_REWARD", "EPISODE_TIMEOUT"):
        assert k in envs
    assert all_bounds()["KILL_REWARD"] == (2.0, 20.0)


def test_clamp_respects_bounds_and_type():
    assert clamp("KILL_REWARD", 99) == 20.0          # float, capped
    assert clamp("KILL_REWARD", -5) == 2.0           # float, floored
    assert clamp("EPISODE_TIMEOUT", 99999) == 8400   # int, capped + rounded
    assert isinstance(clamp("EPISODE_TIMEOUT", 2000.7), int)
    assert clamp("UNKNOWN_KNOB", 5) == 5             # unknown passes through


def test_validate_drops_unknown_and_clamps():
    out = validate({"KILL_REWARD": 50, "BOGUS": 1, "ENGAGEMENT_REWARD": 0.3},
                   base_env={"HIT_REWARD": "3"})
    assert out["KILL_REWARD"] == "20.0"              # clamped to cap
    assert out["ENGAGEMENT_REWARD"] == "0.3"
    assert "BOGUS" not in out                        # hallucinated knob rejected
    assert out["HIT_REWARD"] == "3"                  # base preserved


def test_describe_lists_params_with_current_values():
    text = describe_for_llm({"KILL_REWARD": "8.0"})
    assert "KILL_REWARD = 8.0" in text and "range" in text
    assert "[combat]" in text and "[explore]" in text   # grouped
