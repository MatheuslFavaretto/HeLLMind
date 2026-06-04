"""Tests for rl.experiment — hypothesis-driven A/B experiment engine."""
from types import SimpleNamespace


from rl.experiment import (
    ExperimentPlan,
    ExperimentResult,
    _judge,
    record_result,
    write_experiment_note,
)


def _cfg(tmp_path):
    return SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        vault_path=str(tmp_path / "vault"),
    )


def _plan(metric="map_explored", direction="up", seeds=None):
    return ExperimentPlan(
        hypothesis_id=1,
        title="Test hypothesis",
        metric=metric,
        direction=direction,
        control_env={"COVERAGE_REWARD": "1.0", "FRONTIER_REWARD": "0.0"},
        experimental_env={"COVERAGE_REWARD": "1.0", "FRONTIER_REWARD": "0.02"},
        seeds=seeds or [42, 123],
        steps=200000,
    )


# ---------------------------------------------------------------------------
# _judge
# ---------------------------------------------------------------------------

def test_judge_improved():
    plan = _plan(metric="map_explored", direction="up")
    ctrl = [{"map_explored": 0.10}, {"map_explored": 0.12}]
    exp  = [{"map_explored": 0.20}, {"map_explored": 0.22}]
    verdict, conf, notes = _judge(plan, ctrl, exp)
    assert verdict == "improved"
    assert conf > 0.0
    assert "relative_change" in notes


def test_judge_regressed():
    plan = _plan(metric="map_explored", direction="up")
    ctrl = [{"map_explored": 0.20}]
    exp  = [{"map_explored": 0.10}]
    verdict, conf, notes = _judge(plan, ctrl, exp)
    assert verdict == "regressed"


def test_judge_no_effect():
    plan = _plan(metric="map_explored", direction="up")
    ctrl = [{"map_explored": 0.20}]
    exp  = [{"map_explored": 0.21}]
    verdict, conf, notes = _judge(plan, ctrl, exp)
    assert verdict == "no_effect"


def test_judge_zero_baseline():
    plan = _plan(metric="exit_rate", direction="up")
    ctrl = [{"exit_rate": 0.0}]
    exp  = [{"exit_rate": 0.10}]
    verdict, conf, notes = _judge(plan, ctrl, exp)
    assert verdict == "improved"


def test_judge_direction_down():
    plan = _plan(metric="deaths_per_ep", direction="down")
    ctrl = [{"deaths_per_ep": 5.0}]
    exp  = [{"deaths_per_ep": 3.0}]
    verdict, conf, notes = _judge(plan, ctrl, exp)
    assert verdict == "improved"


def test_judge_confidence_scales_with_seeds():
    plan_1 = _plan(seeds=[42])
    plan_3 = _plan(seeds=[42, 123, 777])
    ctrl  = [{"map_explored": 0.10}]
    exp   = [{"map_explored": 0.20}]
    ctrl3 = ctrl * 3
    exp3  = exp * 3

    _, conf1, _ = _judge(plan_1, ctrl, exp)
    _, conf3, _ = _judge(plan_3, ctrl3, exp3)
    assert conf3 > conf1


# ---------------------------------------------------------------------------
# record_result
# ---------------------------------------------------------------------------

def _result(tmp_path, verdict="improved", confidence=0.7):
    plan = _plan()
    return ExperimentResult(
        plan=plan,
        control_metrics=[{"map_explored": 0.10}],
        experimental_metrics=[{"map_explored": 0.20}],
        verdict=verdict,
        confidence=confidence,
        notes="test notes",
    )


def test_record_result_inserts_experiment(tmp_path):
    from writer.db import insert_hypothesis, query_experiments
    cfg = _cfg(tmp_path)
    # pre-insert hypothesis so update_status finds it
    hid = insert_hypothesis(
        cfg.memory_dir, "Test hypothesis", "body", "map_explored", "up", 0.8
    )
    result = _result(tmp_path)
    result.plan.hypothesis_id = hid
    record_result(cfg, result)

    exps = query_experiments(cfg.memory_dir)
    assert len(exps) == 1
    assert exps[0]["result"] == "improved"


def test_record_result_updates_hypothesis_status(tmp_path):
    from writer.db import insert_hypothesis, query_hypotheses
    cfg = _cfg(tmp_path)
    hid = insert_hypothesis(
        cfg.memory_dir, "Test hypothesis", "body", "map_explored", "up", 0.8
    )
    result = _result(tmp_path, verdict="improved")
    result.plan.hypothesis_id = hid
    record_result(cfg, result)
    rows = query_hypotheses(cfg.memory_dir, status="confirmed")
    assert len(rows) == 1


def test_record_rejected_hypothesis(tmp_path):
    from writer.db import insert_hypothesis, query_hypotheses
    cfg = _cfg(tmp_path)
    hid = insert_hypothesis(
        cfg.memory_dir, "Test hypothesis", "body", "map_explored", "up", 0.8
    )
    result = _result(tmp_path, verdict="regressed")
    result.plan.hypothesis_id = hid
    record_result(cfg, result)
    rows = query_hypotheses(cfg.memory_dir, status="rejected")
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# write_experiment_note
# ---------------------------------------------------------------------------

def test_write_experiment_note(tmp_path):
    cfg = _cfg(tmp_path)
    result = _result(tmp_path)
    path = write_experiment_note(cfg, result)
    assert path.endswith("Experiment-H1.md")
    content = open(path).read()
    assert "improved" in content
    assert "FRONTIER_REWARD" in content
    assert "map_explored" in content


def test_write_experiment_note_regressed(tmp_path):
    cfg = _cfg(tmp_path)
    result = _result(tmp_path, verdict="regressed")
    path = write_experiment_note(cfg, result)
    content = open(path).read()
    assert "regressed" in content
    assert "❌" in content
