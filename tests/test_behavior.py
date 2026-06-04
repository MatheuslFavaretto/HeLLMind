"""Tests for writer.behavior — behavioral flag detectors."""

from writer.behavior import (
    BehaviorFlag,
    detect,
    detect_circling,
    detect_low_exploration,
    detect_passive,
    detect_route_repetition,
    detect_shoot_spam,
)


# ---------------------------------------------------------------------------
# shoot_spam
# ---------------------------------------------------------------------------

def test_shoot_spam_flagged_on_low_accuracy():
    snaps = [{"shooting_accuracy": 0.03}] * 5
    flag = detect_shoot_spam(snaps)
    assert flag is not None
    assert flag.name == "shoot_spam"
    assert flag.confidence > 0.5


def test_shoot_spam_not_flagged_on_decent_accuracy():
    snaps = [{"shooting_accuracy": 0.40}] * 5
    assert detect_shoot_spam(snaps) is None


def test_shoot_spam_empty():
    assert detect_shoot_spam([]) is None
    assert detect_shoot_spam([{"other_key": 1}]) is None


# ---------------------------------------------------------------------------
# low_exploration
# ---------------------------------------------------------------------------

def _event(coverage, etype="timeout", map_name="MAP01"):
    return {"type": etype, "map": map_name, "coverage": coverage}


def test_low_exploration_flagged():
    events = [_event(0.05)] * 10
    flag = detect_low_exploration(events)
    assert flag is not None
    assert flag.name == "low_exploration"
    assert flag.confidence > 0.0


def test_low_exploration_not_flagged():
    events = [_event(0.50)] * 5
    assert detect_low_exploration(events) is None


def test_low_exploration_raw_cell_count():
    # coverage reported as cell count (> 1) — should be normalised
    events = [_event(50)] * 5   # 50/1000 = 5%
    flag = detect_low_exploration(events)
    assert flag is not None


def test_low_exploration_empty():
    assert detect_low_exploration([]) is None
    assert detect_low_exploration([{"type": "death"}]) is None


# ---------------------------------------------------------------------------
# passive
# ---------------------------------------------------------------------------

def _kill_event(kills, etype="timeout"):
    return {"type": etype, "kills": kills, "map": "MAP01"}


def test_passive_flagged():
    events = [_kill_event(0)] * 10
    flag = detect_passive(events, [])
    assert flag is not None
    assert flag.name == "passive"
    assert flag.confidence > 0.5


def test_passive_not_flagged_on_kills():
    events = [_kill_event(3)] * 10
    assert detect_passive(events, []) is None


def test_passive_uses_snapshot_cross_check():
    events = [_kill_event(0)] * 5
    snaps = [{"kills_per_episode": 0.1}] * 3
    flag = detect_passive(events, snaps)
    assert flag is not None


# ---------------------------------------------------------------------------
# circling
# ---------------------------------------------------------------------------

def test_circling_flagged_on_plateau():
    snaps = [{"map_explored": 0.04}] * 6  # no growth
    flag = detect_circling([], snaps)
    assert flag is not None
    assert flag.name == "circling"
    assert flag.confidence >= 0.8


def test_circling_not_flagged_when_growing():
    snaps = [{"map_explored": 0.05}, {"map_explored": 0.15}, {"map_explored": 0.30}]
    assert detect_circling([], snaps) is None


def test_circling_fallback_low_coverage():
    events = [_event(0.03)] * 8
    flag = detect_circling(events, [])  # no snapshots
    assert flag is not None
    assert flag.confidence <= 0.5  # low confidence fallback


# ---------------------------------------------------------------------------
# route_repetition
# ---------------------------------------------------------------------------

def test_route_repetition_flagged():
    events = [_event(0.04, map_name="MAP01")] * 10
    flag = detect_route_repetition(events)
    assert flag is not None
    assert flag.name == "route_repetition"


def test_route_repetition_not_flagged_when_growing():
    events = [
        _event(0.05, map_name="MAP01"),
        _event(0.20, map_name="MAP01"),
        _event(0.50, map_name="MAP01"),
    ]
    assert detect_route_repetition(events) is None


def test_route_repetition_needs_three_events():
    events = [_event(0.04)] * 2
    assert detect_route_repetition(events) is None


# ---------------------------------------------------------------------------
# detect (combined)
# ---------------------------------------------------------------------------

def test_detect_combined_all_bad():
    events = [_event(0.03, map_name="MAP01")] * 10 + [_kill_event(0)] * 10
    snaps = [{"shooting_accuracy": 0.02, "map_explored": 0.03}] * 5
    flags = detect(events, snaps)
    names = {f.name for f in flags}
    assert "low_exploration" in names
    assert "shoot_spam" in names
    assert "passive" in names


def test_detect_empty_data():
    flags = detect([], [])
    assert flags == []


def test_flag_confidence_in_range():
    events = [_event(0.02)] * 20
    snaps = [{"shooting_accuracy": 0.01}] * 5
    for f in detect(events, snaps):
        assert 0.0 <= f.confidence <= 1.0
