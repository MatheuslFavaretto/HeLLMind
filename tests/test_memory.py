"""Cognitive memory: persistent event store (Phase 1) and lesson aggregation (Phase 4)."""
import os

import numpy as np

from writer.memory_store import MemoryStore
from writer.reflect import aggregate_events


def test_record_and_read_events(tmp_path):
    store = MemoryStore(str(tmp_path), run_name="run-A")
    store.record_event({"type": "death", "map": "MAP01", "health": np.int64(12)})
    store.record_event({"type": "success", "map": "MAP01"})
    events = MemoryStore.read_events(str(tmp_path))
    assert len(events) == 2
    assert events[0]["type"] == "death" and events[0]["health"] == 12  # numpy sanitized
    assert events[0]["run"] == "run-A" and "ts" in events[0]


def test_events_persist_across_stores(tmp_path):
    MemoryStore(str(tmp_path), "run-A").record_event({"type": "death"})
    MemoryStore(str(tmp_path), "run-B").record_event({"type": "success"})  # appends
    runs = {e["run"] for e in MemoryStore.read_events(str(tmp_path))}
    assert runs == {"run-A", "run-B"}


def test_read_missing_returns_empty(tmp_path):
    assert MemoryStore.read_events(os.path.join(tmp_path, "nope")) == []


def test_aggregate_events():
    events = (
        [{"type": "death", "map": "MAP01", "health": 10, "ammo": 2, "length": 80}] * 3
        + [{"type": "death", "map": "MAP02", "health": 60, "ammo": 40, "length": 120}]
        + [{"type": "success", "map": "MAP01", "health": 80, "length": 300}]
        * 1
    )
    for e in events:  # stamp runs so the run count is meaningful
        e.setdefault("run", "r1")
    s = aggregate_events(events)
    assert s["total"] == 5
    assert s["deaths"] == 4 and s["successes"] == 1
    assert s["death_rate"] == 0.8
    assert s["low_hp_death_rate"] == 0.75          # 3 of 4 deaths under 30 HP
    assert s["deaths_by_map"] == {"MAP01": 3, "MAP02": 1}
    assert s["mean_len_success"] == 300


def test_aggregate_empty():
    s = aggregate_events([])
    assert s["total"] == 0 and s["death_rate"] == 0.0 and s["deaths_by_map"] == {}
    # exploration keys exist and are zero/empty on no data
    assert s["exit_rate"] == 0.0 and s["mean_coverage"] == 0.0
    assert s["stuck_maps"] == [] and s["coverage_by_map"] == {}


def test_aggregate_exploration_dimensions():
    events = [
        {"type": "exit", "map": "MAP01", "coverage": 90, "length": 250, "run": "r1"},
        {"type": "timeout", "map": "MAP02", "coverage": 8, "length": 400, "run": "r1"},
        {"type": "timeout", "map": "MAP02", "coverage": 12, "length": 400, "run": "r1"},
        {"type": "death", "map": "MAP02", "coverage": 10, "health": 5, "run": "r1"},
    ]
    s = aggregate_events(events)
    assert s["exits"] == 1 and s["exit_rate"] == 0.25
    assert s["completions"] == 1                      # exit counts as a completion
    assert s["timeouts_by_map"] == {"MAP02": 2}
    assert s["coverage_by_map"]["MAP01"] == 90.0
    assert s["coverage_by_map"]["MAP02"] == 10.0       # mean of 8,12,10
    assert s["stuck_maps"] == ["MAP02"]                # 10 << 0.6*90 -> under-explored
    assert s["mean_coverage_exit"] == 90.0


def test_lessons_and_suggest_prompts_mention_exploration():
    from writer.prompts import build_lessons_user_message, build_suggest_user_message
    stats = {"total": 4, "runs": 1, "exits": 1, "exit_rate": 0.25, "mean_coverage": 30.0,
             "stuck_maps": ["MAP02"], "timeouts_by_map": {"MAP02": 2},
             "coverage_by_map": {"MAP01": 90.0, "MAP02": 10.0}}
    lessons = build_lessons_user_message(stats)
    assert "Exits reached" in lessons and "stuck maps" in lessons.lower()
    suggest = build_suggest_user_message(stats, {"hit_reward": 2.0})
    assert "Exit rate" in suggest and "stuck maps" in suggest.lower()
