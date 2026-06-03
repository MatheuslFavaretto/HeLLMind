"""Tests for rl.research_agent — the meta-loop orchestrator."""
import json
import os
from types import SimpleNamespace

import pytest

from rl.research_agent import ResearchIteration, _score, _write_log


def _cfg(tmp_path):
    return SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        vault_path=str(tmp_path / "vault"),
        pending_dir=str(tmp_path / "pending"),
        dir_index="00-index",
        maps=["MAP01", "MAP02"],
        run_name="run-test",
    )


def _iter(i, verdict=None, score_before=0.5, score_after=0.7):
    return ResearchIteration(
        iteration=i,
        ts="2025-01-01T00:00:00+00:00",
        flags=["low_exploration"],
        hypotheses=["Raising FRONTIER_REWARD will push exploration"],
        experiment_verdict=verdict,
        experiment_metric="map_explored" if verdict else None,
        experiment_param='{"FRONTIER_REWARD": "0.03"}' if verdict else None,
        score_before=score_before,
        score_after=score_after,
        curriculum_weights={"MAP01": 1.2, "MAP02": 0.8},
        notes="test iteration",
    )


# ---------------------------------------------------------------------------
# _score
# ---------------------------------------------------------------------------

def test_score_all_zeros():
    assert _score({}) == pytest.approx(0.0)


def test_score_perfect_agent():
    m = {"exit_rate": 1.0, "explored_fraction": 1.0,
         "kills_per_episode": 10.0, "shooting_accuracy": 1.0}
    s = _score(m)
    assert s > 10.0


def test_score_weights_exit_highest():
    exit_agent  = {"exit_rate": 1.0, "explored_fraction": 0.0, "kills_per_episode": 0, "shooting_accuracy": 0}
    expl_agent  = {"exit_rate": 0.0, "explored_fraction": 1.0, "kills_per_episode": 0, "shooting_accuracy": 0}
    assert _score(exit_agent) > _score(expl_agent)


# ---------------------------------------------------------------------------
# _write_log
# ---------------------------------------------------------------------------

def test_write_log_creates_jsonl(tmp_path):
    cfg = _cfg(tmp_path)
    history = [_iter(0, verdict=None), _iter(1, verdict="improved")]
    _write_log(cfg, history)
    path = os.path.join(cfg.memory_dir, "research_log.jsonl")
    assert os.path.exists(path)
    lines = [json.loads(l) for l in open(path)]
    assert len(lines) == 2
    assert lines[0]["iteration"] == 0
    assert lines[1]["experiment_verdict"] == "improved"


def test_write_log_creates_vault_note(tmp_path):
    cfg = _cfg(tmp_path)
    history = [_iter(0)]
    _write_log(cfg, history)
    note = os.path.join(cfg.vault_path, "00-index", "Research Log.md")
    assert os.path.exists(note)
    content = open(note).read()
    assert "Research Agent Log" in content
    assert "low_exploration" in content


def test_write_log_milestone_detected(tmp_path):
    cfg = _cfg(tmp_path)
    history = [_iter(0, verdict="improved", score_before=0.5, score_after=0.9)]
    _write_log(cfg, history)
    note = os.path.join(cfg.vault_path, "00-index", "Research Log.md")
    content = open(note).read()
    assert "MILESTONE" in content


def test_write_log_no_milestone(tmp_path):
    cfg = _cfg(tmp_path)
    history = [_iter(0, verdict="regressed")]
    _write_log(cfg, history)
    note = os.path.join(cfg.vault_path, "00-index", "Research Log.md")
    content = open(note).read()
    assert "Milestone not yet hit" in content


def test_write_log_idempotent(tmp_path):
    cfg = _cfg(tmp_path)
    history = [_iter(0)]
    _write_log(cfg, history)
    _write_log(cfg, history)  # second write replaces, no duplicates
    lines = open(os.path.join(cfg.memory_dir, "research_log.jsonl")).readlines()
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# ResearchIteration fields
# ---------------------------------------------------------------------------

def test_iteration_serializable():
    it = _iter(0, verdict="no_effect")
    d = it.__dict__
    json.dumps(d)  # must not raise


def test_score_delta_in_notes_direction(tmp_path):
    cfg = _cfg(tmp_path)
    history = [_iter(0, score_before=1.0, score_after=2.0)]
    _write_log(cfg, history)
    note = open(os.path.join(cfg.vault_path, "00-index", "Research Log.md")).read()
    assert "+1.00" in note or "1.00" in note
