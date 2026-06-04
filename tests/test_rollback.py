"""Tests for the structured rollback log (P4 — never degrade permanently)."""
from writer.rollback import RollbackLog, diff_envs


def test_diff_envs_only_changed_knobs():
    before, change, after = diff_envs(
        {"COVERAGE_REWARD": "1.0", "HIT_REWARD": "3.0"},
        {"COVERAGE_REWARD": "1.4", "HIT_REWARD": "3.0"},
    )
    assert change == {"COVERAGE_REWARD": ["1.0", "1.4"]}
    assert before == {"COVERAGE_REWARD": "1.0"}
    assert after == {"COVERAGE_REWARD": "1.4"}


def test_record_and_history(tmp_path):
    log = RollbackLog(str(tmp_path))
    log.record(1, {"COVERAGE_REWARD": "1.0"}, {"COVERAGE_REWARD": ["1.0", "1.4"]},
               {"COVERAGE_REWARD": "1.4"}, {"score": 0.55}, kept=True)
    log.record(2, {"COVERAGE_REWARD": "1.4"}, {"COVERAGE_REWARD": ["1.4", "1.8"]},
               {"COVERAGE_REWARD": "1.8"}, {"score": 0.30}, kept=False)
    hist = log.history()
    assert len(hist) == 2
    assert hist[0]["before"] == {"COVERAGE_REWARD": "1.0"}
    assert hist[0]["result"]["score"] == 0.55
    assert hist[1]["kept"] is False


def test_reverted_filters_regressions(tmp_path):
    log = RollbackLog(str(tmp_path))
    log.record(1, {}, {"A": [1, 2]}, {}, {"score": 0.5}, kept=True)
    log.record(2, {}, {"A": [2, 3]}, {}, {"score": 0.2}, kept=False)
    rev = log.reverted()
    assert len(rev) == 1 and rev[0]["iter"] == 2


def test_empty_history(tmp_path):
    assert RollbackLog(str(tmp_path)).history() == []
