"""Tests for the long-term knowledge tiers (facts / hypotheses / validated)."""
from writer.bestiary import BestiaryStore
from writer.db import insert_experiment, insert_hypothesis, update_hypothesis_status
from writer.knowledge import knowledge_tiers
from writer.learned_config import LearnedConfig


def test_empty_memory_has_all_tiers(tmp_path):
    t = knowledge_tiers(str(tmp_path))
    assert set(t) == {"facts", "hypotheses", "validated"}
    assert t["facts"] == [] and t["validated"] == []


def test_bestiary_becomes_facts(tmp_path):
    BestiaryStore(str(tmp_path)).merge({
        "DoomImp": {"encounters": 5, "killed_agent": 3, "killed": 1, "ranged": True},
    })
    facts = knowledge_tiers(str(tmp_path))["facts"]
    assert len(facts) == 1
    assert "Imp" in facts[0]["text"]
    assert "killed the agent 3" in facts[0]["text"]


def test_open_hypothesis_is_in_hypotheses_tier(tmp_path):
    insert_hypothesis(str(tmp_path), "Raise frontier", "body", "map_explored", "up", 0.8)
    t = knowledge_tiers(str(tmp_path))
    assert any("frontier" in h["text"].lower() for h in t["hypotheses"])
    assert t["validated"] == []


def test_confirmed_hypothesis_and_experiment_are_validated(tmp_path):
    hid = insert_hypothesis(str(tmp_path), "Raise coverage", "b", "explored", "up", 0.7)
    update_hypothesis_status(str(tmp_path), hid, "confirmed")
    insert_experiment(str(tmp_path), "COVERAGE_REWARD", "1.0", "2.0", "improved", 0.9)
    validated = knowledge_tiers(str(tmp_path))["validated"]
    texts = " ".join(v["text"] for v in validated)
    assert "Raise coverage" in texts
    assert "COVERAGE_REWARD" in texts


def test_learned_config_is_validated(tmp_path):
    LearnedConfig(str(tmp_path)).adopt({"HIT_REWARD": 3.0}, source="experiment")
    validated = knowledge_tiers(str(tmp_path))["validated"]
    assert any("HIT_REWARD" in v["text"] for v in validated)
