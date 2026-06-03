"""Tests for rl.curriculum — difficulty scoring + forgetting detection."""
import pytest

from rl.curriculum import (
    MapDifficulty,
    ForgettingAlert,
    detect_forgetting,
    difficulty_score,
    difficulty_weights,
    smart_weights,
    write_curriculum_note,
)


def _events(map_name, n_deaths=0, n_timeouts=0, n_exits=0, coverage=50, kills=3):
    evts = []
    for _ in range(n_deaths):
        evts.append({"type": "death",   "map": map_name, "coverage": coverage, "kills": kills})
    for _ in range(n_timeouts):
        evts.append({"type": "timeout", "map": map_name, "coverage": coverage, "kills": kills})
    for _ in range(n_exits):
        evts.append({"type": "exit",    "map": map_name, "coverage": coverage, "kills": kills})
    return evts


# ---------------------------------------------------------------------------
# difficulty_score
# ---------------------------------------------------------------------------

def test_difficulty_score_all_deaths():
    events = _events("MAP01", n_deaths=10)
    d = difficulty_score("MAP01", events, coverage_scale=200)
    assert d is not None
    assert d.death_rate == pytest.approx(1.0)
    assert d.score > 2.0   # high difficulty


def test_difficulty_score_all_exits_high_coverage():
    events = _events("MAP01", n_exits=10, coverage=200, kills=5)
    d = difficulty_score("MAP01", events, coverage_scale=200)
    assert d is not None
    assert d.death_rate == pytest.approx(0.0)
    assert d.score < 1.0   # easy map


def test_difficulty_score_no_events():
    assert difficulty_score("MAP02", [], coverage_scale=200) is None


def test_difficulty_score_fields():
    events = _events("MAP03", n_deaths=5, n_timeouts=5, coverage=80, kills=2)
    d = difficulty_score("MAP03", events, coverage_scale=200)
    assert d.map_name == "MAP03"
    assert d.n_episodes == 10
    assert 0.0 <= d.score <= 4.0


# ---------------------------------------------------------------------------
# difficulty_weights
# ---------------------------------------------------------------------------

def test_difficulty_weights_normalised():
    events = (
        _events("MAP01", n_deaths=8, n_timeouts=2, coverage=20, kills=1)
        + _events("MAP02", n_exits=10, coverage=180, kills=5)
    )
    maps = ["MAP01", "MAP02"]
    w = difficulty_weights(events, maps, coverage_scale=200)
    mean = sum(w.values()) / len(maps)
    assert mean == pytest.approx(1.0, abs=0.01)
    assert w["MAP01"] > w["MAP02"]  # harder map gets more budget


def test_difficulty_weights_no_data_defaults_to_one():
    w = difficulty_weights([], ["MAP01", "MAP02"])
    assert all(abs(v - 1.0) < 0.01 for v in w.values())


# ---------------------------------------------------------------------------
# detect_forgetting
# ---------------------------------------------------------------------------

def _forgetting_events(map_name, good_n=30, bad_n=25, good_kills=4.0, bad_kills=1.0):
    evts = []
    for _ in range(good_n):
        evts.append({"type": "exit", "map": map_name, "kills": good_kills, "coverage": 100})
    for _ in range(bad_n):
        evts.append({"type": "timeout", "map": map_name, "kills": bad_kills, "coverage": 20})
    return evts


def test_detect_forgetting_detects_drop():
    events = _forgetting_events("MAP01", good_kills=4.0, bad_kills=0.5)
    alerts = detect_forgetting(events, ["MAP01"], threshold=0.30, window=10)
    assert len(alerts) > 0
    assert any(a.metric == "kills" for a in alerts)
    assert any(a.map_name == "MAP01" for a in alerts)


def test_detect_forgetting_no_drop():
    events = _events("MAP01", n_exits=60, kills=4, coverage=100)
    alerts = detect_forgetting(events, ["MAP01"], threshold=0.30, window=10)
    assert alerts == []


def test_detect_forgetting_insufficient_history():
    events = _events("MAP01", n_exits=5, kills=4)
    alerts = detect_forgetting(events, ["MAP01"], threshold=0.30, window=10)
    assert alerts == []   # needs window*2 events minimum


def test_detect_forgetting_drop_fraction():
    events = _forgetting_events("MAP02", good_kills=5.0, bad_kills=1.0)
    alerts = detect_forgetting(events, ["MAP02"], threshold=0.30, window=10)
    kill_alerts = [a for a in alerts if a.metric == "kills"]
    if kill_alerts:
        a = kill_alerts[0]
        assert a.drop_fraction >= 0.30
        assert a.peak_value > a.recent_value


# ---------------------------------------------------------------------------
# smart_weights
# ---------------------------------------------------------------------------

def test_smart_weights_boosts_regressed_maps():
    events = (
        _forgetting_events("MAP01", good_kills=4.0, bad_kills=0.5)    # regressing
        + _events("MAP02", n_exits=60, kills=4, coverage=100)         # stable
    )
    maps = ["MAP01", "MAP02"]
    w = smart_weights(events, maps)
    # When MAP01 is regressing it should get more budget than stable MAP02
    assert isinstance(w, dict) and len(w) == 2


def test_smart_weights_normalised():
    events = (
        _events("MAP01", n_deaths=10, coverage=20, kills=1)
        + _events("MAP02", n_exits=10, coverage=180, kills=5)
    )
    maps = ["MAP01", "MAP02"]
    w = smart_weights(events, maps)
    mean = sum(w.values()) / len(maps)
    assert mean == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# write_curriculum_note
# ---------------------------------------------------------------------------

def test_write_curriculum_note(tmp_path):
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        vault_path=str(tmp_path),
        memory_dir=str(tmp_path / "memory"),
        dir_maps="40-maps",
        maps=["MAP01", "MAP02"],
    )
    events = (
        _events("MAP01", n_deaths=5, n_timeouts=5, coverage=30, kills=2)
        + _events("MAP02", n_exits=10, coverage=150, kills=4)
    )
    path = write_curriculum_note(cfg, ["MAP01", "MAP02"], events)
    assert path.endswith("Curriculum.md")
    content = open(path).read()
    assert "MAP01" in content and "MAP02" in content
    assert "Score" in content
