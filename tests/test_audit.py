"""Tests for rl.audit — the RL quality grader (no TensorBoard needed; we feed series in)."""
from rl.audit import _last, _trend, grade


def _series(values):
    """Turn a list of values into the (step, value) pairs audit expects."""
    return [(i * 1000, v) for i, v in enumerate(values)]


# --------------------------- helpers ---------------------------

def test_last_averages_tail():
    s = _series([0, 0, 10, 10, 10])
    assert _last(s, n=3) == 10.0


def test_last_empty_is_none():
    assert _last([]) is None


def test_trend_positive_when_rising():
    assert _trend(_series([1, 2, 3, 4, 5])) > 0


def test_trend_negative_when_falling():
    assert _trend(_series([5, 4, 3, 2, 1])) < 0


def test_trend_too_few_points_is_none():
    assert _trend(_series([1])) is None


# --------------------------- grade() ---------------------------

def test_grade_healthy_run_scores_high():
    # _last() averages the tail (last 5), so feed a stable high-EV tail.
    data = {
        "ev":       _series([0.85, 0.86, 0.87, 0.88, 0.9]),  # tail mean ≥ 0.8
        "entropy":  _series([-1.0, -1.1, -1.2, -1.3, -1.4]),  # gentle decline
        "kl":       _series([0.005, 0.006, 0.005, 0.005, 0.006]),
        "val_loss": _series([20, 16, 12, 10, 8]),            # falling
        "ep_rew":   _series([10, 20, 30, 40, 50]),           # rising
    }
    result = grade(data)
    assert result["overall"] is not None
    assert result["overall"] >= 7.0
    assert result["checks"]["value_quality"]["score"] == 9


def test_grade_weak_value_function():
    data = {"ev": _series([0.1, 0.2, 0.3])}
    result = grade(data)
    assert result["checks"]["value_quality"]["score"] == 2


def test_grade_unstable_kl_flagged():
    data = {"kl": _series([0.08, 0.09, 0.1])}
    result = grade(data)
    assert result["checks"]["kl_stability"]["score"] <= 3


def test_grade_falling_reward_flagged():
    data = {"ep_rew": _series([50, 40, 30, 20, 10])}
    result = grade(data)
    assert result["checks"]["reward_learning"]["score"] <= 2


def test_grade_empty_data_no_crash():
    result = grade({})
    # Every check should be N/A, overall None — but it must not raise.
    assert result["overall"] is None
    assert all(c["score"] is None for c in result["checks"].values())
