"""Tests for writer.db (SQLite cognitive memory) and writer.recall (retrieval API)."""
import json
import os

import pytest

from writer.db import (
    build,
    connect,
    insert_experiment,
    insert_hypothesis,
    query_events,
    query_experiments,
    query_hypotheses,
    query_lessons,
    query_maps,
    update_hypothesis_status,
)
from writer.memory_store import MemoryStore
from writer.recall import recall, recall_deaths, recall_enemy, recall_map, recall_region


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem(tmp_path):
    """Return a memory_dir with some pre-populated JSONL data."""
    memory_dir = str(tmp_path)
    store = MemoryStore(memory_dir, run_name="run-A")
    store.record_event({"type": "death", "map": "MAP01", "health": 10.0,
                        "ammo": 5.0, "kills": 1, "coverage": 8.0, "length": 300,
                        "weapon": 1, "region": "0x0", "nearest_enemy": "DoomImp"})
    store.record_event({"type": "death", "map": "MAP01", "health": 25.0,
                        "ammo": 0.0, "kills": 2, "coverage": 9.0, "length": 250,
                        "weapon": 2, "region": "1x0", "nearest_enemy": "Shotguy"})
    store.record_event({"type": "exit",  "map": "MAP01", "health": 80.0,
                        "ammo": 30.0, "kills": 5, "coverage": 45.0, "length": 900,
                        "weapon": 3, "region": "2x1", "nearest_enemy": ""})
    store.record_event({"type": "timeout", "map": "MAP02", "health": 60.0,
                        "ammo": 10.0, "kills": 0, "coverage": 4.0, "length": 1050,
                        "weapon": 1, "region": "0x0", "nearest_enemy": "DoomImp"})

    # Lessons JSONL
    lessons_dir = os.path.join(memory_dir, "lessons")
    os.makedirs(lessons_dir, exist_ok=True)
    with open(os.path.join(lessons_dir, "lessons.jsonl"), "w") as f:
        f.write(json.dumps({"ts": "2025-01-01T00:00:00+00:00", "run": "run-A",
                             "title": "Agent dies in corridors",
                             "insight": "Low HP at death suggests corridors are dangerous.",
                             "evidence": "3 of 4 deaths below 30 HP"}) + "\n")

    # Coverage JSON
    cov_dir = os.path.join(memory_dir, "coverage")
    os.makedirs(cov_dir, exist_ok=True)
    with open(os.path.join(cov_dir, "MAP01.json"), "w") as f:
        json.dump({"map": "MAP01", "cell": 96.0, "runs": 1,
                   "updated": "2025-01-01T00:00:00+00:00",
                   "cells": {"0,0": 5, "1,0": 3},
                   "walls": [[0, 0, 100, 0]]}, f)
    return memory_dir


# ---------------------------------------------------------------------------
# db.build
# ---------------------------------------------------------------------------

def test_build_returns_row_count(mem):
    n = build(mem)
    assert n >= 6  # 4 events + 1 lesson + 1 map


def test_build_populates_events(mem):
    build(mem)
    rows = query_events(mem)
    assert len(rows) == 4
    types = {r["type"] for r in rows}
    assert types == {"death", "exit", "timeout"}


def test_build_populates_lessons(mem):
    build(mem)
    rows = query_lessons(mem)
    assert len(rows) == 1
    assert "corridors" in rows[0]["title"].lower()


def test_build_populates_maps(mem):
    build(mem)
    rows = query_maps(mem)
    assert len(rows) == 1
    assert rows[0]["map"] == "MAP01"
    assert rows[0]["runs"] == 1


def test_build_idempotent(mem):
    build(mem)
    build(mem)  # second build should not duplicate
    assert len(query_events(mem)) == 4
    assert len(query_lessons(mem)) == 1


# ---------------------------------------------------------------------------
# query_events filters
# ---------------------------------------------------------------------------

def test_query_events_by_type(mem):
    build(mem)
    deaths = query_events(mem, event_type="death")
    assert all(r["type"] == "death" for r in deaths)
    assert len(deaths) == 2


def test_query_events_by_map(mem):
    build(mem)
    rows = query_events(mem, map_name="MAP02")
    assert len(rows) == 1
    assert rows[0]["map"] == "MAP02"


def test_query_events_by_type_and_map(mem):
    build(mem)
    rows = query_events(mem, event_type="death", map_name="MAP01")
    assert len(rows) == 2


def test_query_events_empty_db(tmp_path):
    build(str(tmp_path))
    assert query_events(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# query_lessons filters
# ---------------------------------------------------------------------------

def test_query_lessons_keyword(mem):
    build(mem)
    rows = query_lessons(mem, keyword="corridor")
    assert len(rows) == 1

    none = query_lessons(mem, keyword="nonexistent_xyz")
    assert none == []


# ---------------------------------------------------------------------------
# Hypotheses & experiments
# ---------------------------------------------------------------------------

def test_insert_and_query_hypothesis(tmp_path):
    mem = str(tmp_path)
    hid = insert_hypothesis(
        mem,
        title="Higher frontier reward will increase exploration",
        body="Exploration is stuck at 9%. Raising FRONTIER_REWARD should push past 25%.",
        metric="map_explored",
        direction="up",
        confidence=0.7,
    )
    assert hid == 1
    rows = query_hypotheses(mem)
    assert len(rows) == 1
    assert rows[0]["status"] == "open"
    assert rows[0]["confidence"] == pytest.approx(0.7)


def test_update_hypothesis_status(tmp_path):
    mem = str(tmp_path)
    hid = insert_hypothesis(mem, "Test", "body", "kills", "up", 0.5)
    update_hypothesis_status(mem, hid, "confirmed")
    rows = query_hypotheses(mem, status="confirmed")
    assert len(rows) == 1
    assert rows[0]["id"] == hid


def test_insert_and_query_experiment(tmp_path):
    mem = str(tmp_path)
    eid = insert_experiment(
        mem,
        param="FRONTIER_REWARD",
        old_val="0.0",
        new_val="0.02",
        result="regressed",
        confidence=0.8,
        notes="exploration dropped 9%→2%, kills 4→0",
    )
    assert eid == 1
    rows = query_experiments(mem)
    assert len(rows) == 1
    assert rows[0]["result"] == "regressed"
    assert rows[0]["param"] == "FRONTIER_REWARD"


# ---------------------------------------------------------------------------
# recall API
# ---------------------------------------------------------------------------

def test_recall_returns_lessons_and_events(mem):
    build(mem)
    results = recall("deaths on MAP01", memory_dir=mem)
    assert len(results) > 0
    sources = {r["source"] for r in results}
    assert "lesson" in sources or "event" in sources


def test_recall_map(mem):
    build(mem)
    rows = recall_map("MAP02", memory_dir=mem)
    assert all(r["map"] == "MAP02" for r in rows)
    assert len(rows) == 1


def test_recall_deaths_low_hp(mem):
    build(mem)
    rows = recall_deaths(memory_dir=mem, max_hp=30.0)
    assert all(r["health"] <= 30.0 for r in rows)
    assert len(rows) == 2  # health=10 and health=25


def test_recall_deaths_min_hp(mem):
    build(mem)
    rows = recall_deaths(memory_dir=mem, min_hp=50.0)
    assert all(r["health"] >= 50.0 for r in rows)


def test_recall_empty_db(tmp_path):
    results = recall("MAP01", memory_dir=str(tmp_path))
    assert results == []


# ---------------------------------------------------------------------------
# Episodic context fields (weapon, region, nearest_enemy)
# ---------------------------------------------------------------------------

def test_events_have_weapon_field(mem):
    build(mem)
    rows = query_events(mem, map_name="MAP01")
    weapons = [r.get("weapon") for r in rows]
    assert any(w is not None for w in weapons)


def test_events_have_region_field(mem):
    build(mem)
    rows = query_events(mem, map_name="MAP01")
    regions = [r.get("region") for r in rows]
    assert any(r for r in regions)


def test_events_have_nearest_enemy_field(mem):
    build(mem)
    rows = query_events(mem, event_type="death", map_name="MAP01")
    enemies = [r.get("nearest_enemy") for r in rows]
    assert "DoomImp" in enemies or "Shotguy" in enemies


def test_recall_enemy_finds_imp(mem):
    build(mem)
    rows = recall_enemy("DoomImp", memory_dir=mem)
    assert len(rows) >= 1
    assert all("DoomImp" in (r.get("nearest_enemy") or "") for r in rows)


def test_recall_enemy_case_partial(mem):
    build(mem)
    rows = recall_enemy("shotguy", memory_dir=mem)
    # Partial match is case-insensitive (LIKE) — SQLite LIKE is case-insensitive for ASCII
    assert len(rows) >= 1


def test_recall_region_finds_events(mem):
    build(mem)
    rows = recall_region("0x0", memory_dir=mem)
    assert len(rows) >= 1
    assert all(r.get("region") == "0x0" for r in rows)


def test_recall_body_includes_context(mem):
    build(mem)
    results = recall("deaths MAP01", memory_dir=mem)
    event_bodies = [r["body"] for r in results if r["source"] == "event"]
    # At least one death event body should mention region or nearest_enemy
    assert any("region=" in b or "near=" in b for b in event_bodies)
