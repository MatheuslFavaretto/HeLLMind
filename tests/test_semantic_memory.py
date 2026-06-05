"""Tests for writer.semantic_memory (V2 Phase 3 — vector DB)."""
import os
import tempfile

import pytest

from writer.semantic_memory import SemanticMemory, index_from_memory_store


@pytest.fixture
def mem(tmp_path):
    m = SemanticMemory(str(tmp_path))
    yield m
    m.close()


def test_add_and_count(mem):
    assert mem.count() == 0
    mem.add("agent died at low HP near DoomImp", meta={"map": "MAP01"})
    assert mem.count() == 1
    mem.add("explored the main hall, found shotgun")
    assert mem.count() == 2


def test_search_returns_top_k(mem):
    texts = [
        "died at low HP in corridor near DoomImp",
        "killed revenant in large room",
        "found blue key near exit",
        "explored 20% of the map",
        "death by fireball from CacoDemon",
    ]
    for t in texts:
        mem.add(t)
    results = mem.search("low health death near monster", top_k=3)
    assert len(results) == 3
    for text, meta, score in results:
        assert isinstance(text, str)
        assert isinstance(meta, dict)
        assert isinstance(score, float)


def test_search_returns_fewer_when_db_small(mem):
    mem.add("one entry")
    results = mem.search("any query", top_k=10)
    assert len(results) == 1


def test_search_empty_db_returns_empty(mem):
    results = mem.search("anything")
    assert results == []


def test_add_batch(mem):
    texts = ["event one", "event two", "event three"]
    ids = mem.add_batch(texts)
    assert len(ids) == 3
    assert mem.count() == 3


def test_meta_round_trip(mem):
    meta = {"map": "MAP02", "terminal": "death", "nearest_enemy": "DoomImp"}
    mem.add("died in MAP02", meta=meta)
    results = mem.search("death", top_k=1)
    assert results[0][1]["map"] == "MAP02"
    assert results[0][1]["nearest_enemy"] == "DoomImp"


def test_index_from_memory_store_empty(tmp_path):
    n = index_from_memory_store(str(tmp_path))
    assert n == 0
    m = SemanticMemory(str(tmp_path))
    assert m.count() == 0
    m.close()


def test_index_from_memory_store_with_events(tmp_path):
    import json
    events = [
        {"map": "MAP01", "terminal": "death", "nearest_enemy": "DoomImp", "health": 10.0},
        {"map": "MAP01", "terminal": "timeout", "region": "1x2", "weapon": "shotgun"},
    ]
    with open(os.path.join(str(tmp_path), "memory.jsonl"), "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    n = index_from_memory_store(str(tmp_path))
    assert n == 2

    m = SemanticMemory(str(tmp_path))
    assert m.count() == 2
    m.close()


def test_algo_class_from_path_detects_qrdqn():
    from rl.algo import algo_class_from_path
    from sb3_contrib import QRDQN
    cls = algo_class_from_path("checkpoints/qrdqn_campaign_a11_final.zip")
    assert cls is QRDQN


def test_algo_class_from_path_detects_ppo():
    from rl.algo import algo_class_from_path
    from stable_baselines3 import PPO
    cls = algo_class_from_path("checkpoints/ppo_campaign_a11_final.zip")
    assert cls is PPO


def test_algo_class_from_path_detects_lstm():
    from rl.algo import algo_class_from_path
    from sb3_contrib import RecurrentPPO
    cls = algo_class_from_path("checkpoints/ppo_campaign_a11_lstm_final.zip")
    assert cls is RecurrentPPO
