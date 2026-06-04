"""Tests for writer.hypothesize — behavior flags → hypotheses."""
import pytest

from writer.behavior import BehaviorFlag
from writer.hypothesize import generate, save_hypotheses, write_hypotheses_note


def _flag(name: str, confidence: float = 0.8) -> BehaviorFlag:
    return BehaviorFlag(
        name=name,
        confidence=confidence,
        description=f"Test flag for {name}",
        evidence="test evidence",
        recommendation="test recommendation",
    )


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_known_flags():
    flags = [_flag("shoot_spam"), _flag("low_exploration"), _flag("passive")]
    hypotheses = generate(flags)
    assert len(hypotheses) == 3
    names = {h.source_flag for h in hypotheses}
    assert names == {"shoot_spam", "low_exploration", "passive"}


def test_generate_unknown_flag_ignored():
    flags = [_flag("unknown_behavior")]
    assert generate(flags) == []


def test_generate_empty():
    assert generate([]) == []


def test_confidence_propagated():
    flags = [_flag("circling", confidence=0.65)]
    h = generate(flags)[0]
    assert h.confidence == pytest.approx(0.65)


def test_hypothesis_has_config_delta():
    h = generate([_flag("shoot_spam")])[0]
    assert "MISS_PENALTY" in h.config_delta
    assert float(h.config_delta["MISS_PENALTY"]) > 0


def test_circling_hypothesis():
    h = generate([_flag("circling")])[0]
    assert h.metric == "map_explored"
    assert h.direction == "up"
    assert "FRONTIER_REWARD" in h.config_delta


def test_passive_hypothesis():
    h = generate([_flag("passive")])[0]
    assert h.metric == "kills_per_episode"
    assert h.direction == "up"
    assert "KILL_REWARD" in h.config_delta


# ---------------------------------------------------------------------------
# save_hypotheses (SQLite)
# ---------------------------------------------------------------------------

def _cfg(tmp_path):
    """Return a minimal Config-like object pointing at tmp_path."""
    from types import SimpleNamespace
    return SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        vault_path=str(tmp_path / "vault"),
    )


def test_save_hypotheses_inserts_rows(tmp_path):
    flags = [_flag("shoot_spam"), _flag("passive")]
    hypotheses = generate(flags)
    cfg = _cfg(tmp_path)
    ids = save_hypotheses(cfg, hypotheses)
    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)


def test_save_hypotheses_queryable(tmp_path):
    from writer import db as _db
    flags = [_flag("low_exploration", confidence=0.9)]
    hypotheses = generate(flags)
    cfg = _cfg(tmp_path)
    save_hypotheses(cfg, hypotheses)
    rows = _db.query_hypotheses(cfg.memory_dir, status="open")
    assert len(rows) == 1
    assert "exploration" in rows[0]["title"].lower()


# ---------------------------------------------------------------------------
# write_hypotheses_note (vault)
# ---------------------------------------------------------------------------

def test_write_hypotheses_note(tmp_path):
    flags = [_flag("circling", 0.7), _flag("shoot_spam", 0.5)]
    hypotheses = generate(flags)
    cfg = _cfg(tmp_path)
    path = write_hypotheses_note(cfg, hypotheses)
    assert path.endswith("Hypotheses.md")
    content = open(path).read()
    assert "circling" in content.lower() or "frontier" in content.lower()
    assert "FRONTIER_REWARD" in content or "MISS_PENALTY" in content


def test_write_hypotheses_note_empty(tmp_path):
    cfg = _cfg(tmp_path)
    path = write_hypotheses_note(cfg, [])
    assert "Hypotheses.md" in path
