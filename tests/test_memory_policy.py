"""Tests for the memory→agent feedback loop: learned_config + memory_policy."""
import json

from writer.learned_config import LearnedConfig
from writer.memory_policy import (
    adopt_improved_experiments,
    death_pattern,
    failed_params,
    propose_from_memory,
)


# --------------------------- LearnedConfig ---------------------------

def test_learned_config_empty(tmp_path):
    lc = LearnedConfig(str(tmp_path))
    assert lc.load() == {}
    assert lc.values() == {}


def test_learned_config_adopt_and_values(tmp_path):
    lc = LearnedConfig(str(tmp_path))
    lc.adopt({"COVERAGE_REWARD": 2.5}, source="experiment H1", verdict="improved", confidence=0.8)
    assert lc.values() == {"COVERAGE_REWARD": "2.5"}
    rec = lc.load()["COVERAGE_REWARD"]
    assert rec["source"] == "experiment H1" and rec["verdict"] == "improved"


def test_learned_config_newer_supersedes(tmp_path):
    lc = LearnedConfig(str(tmp_path))
    lc.adopt({"COVERAGE_REWARD": 2.0}, source="H1")
    lc.adopt({"COVERAGE_REWARD": 3.0}, source="H2")
    assert lc.values()["COVERAGE_REWARD"] == "3.0"


def test_learned_config_apply_to_env_overlays(tmp_path):
    lc = LearnedConfig(str(tmp_path))
    lc.adopt({"COVERAGE_REWARD": 2.5}, source="H1")
    env = {"COVERAGE_REWARD": "1.5", "EXIT_REWARD": "1000"}
    out = lc.apply_to_env(env)
    assert out["COVERAGE_REWARD"] == "2.5"   # learned wins
    assert out["EXIT_REWARD"] == "1000"      # untouched
    assert env["COVERAGE_REWARD"] == "1.5"   # input not mutated


def test_learned_config_corrupt_file_safe(tmp_path):
    lc = LearnedConfig(str(tmp_path))
    with open(lc.path, "w") as f:
        f.write("{ not json")
    assert lc.load() == {}


# --------------------------- death_pattern ---------------------------

def test_death_pattern_empty():
    dp = death_pattern([])
    assert dp["n"] == 0 and dp["low_hp_fraction"] == 0.0


def test_death_pattern_low_hp_and_mode():
    events = [
        {"type": "death", "health": 10, "region": "2x3", "nearest_enemy": "DoomImp"},
        {"type": "death", "health": 25, "region": "2x3", "nearest_enemy": "DoomImp"},
        {"type": "death", "health": 80, "region": "1x1", "nearest_enemy": "Shotguy"},
        {"type": "exit",  "health": 90, "region": "5x5"},   # ignored (not a death)
    ]
    dp = death_pattern(events)
    assert dp["n"] == 3
    assert abs(dp["low_hp_fraction"] - 2 / 3) < 1e-9
    assert dp["top_region"] == "2x3"
    assert dp["top_enemy"] == "DoomImp"


# --------------------------- failed_params ---------------------------

def test_failed_params_collects_regressed_and_no_effect():
    exps = [  # newest first
        {"param": json.dumps({"COVERAGE_REWARD": "3.0"}), "result": "regressed"},
        {"param": json.dumps({"DEATH_PENALTY": "8.0"}), "result": "no_effect"},
        {"param": json.dumps({"HIT_REWARD": "4.0"}), "result": "improved"},  # not avoided
    ]
    avoid = failed_params(exps)
    assert avoid.get("COVERAGE_REWARD") == "regressed"
    assert avoid.get("DEATH_PENALTY") == "no_effect"
    assert "HIT_REWARD" not in avoid


# --------------------------- propose_from_memory ---------------------------

def _deaths(n, health):
    return [{"type": "death", "health": health, "region": "1x1",
             "nearest_enemy": "DoomImp"} for _ in range(n)]


def test_propose_targets_low_hp_deaths():
    events = _deaths(12, health=10)        # all low-HP deaths
    env = {"DAMAGE_TAKEN_PENALTY": "0.1"}
    new, reason = propose_from_memory(events, [], env)
    assert new is not None
    assert float(new["DAMAGE_TAKEN_PENALTY"]) > 0.1
    assert "DAMAGE_TAKEN_PENALTY" in reason


def test_propose_respects_past_failed_experiment():
    events = _deaths(12, health=10)
    # a past experiment already proved raising DAMAGE_TAKEN_PENALTY doesn't help
    exps = [{"param": json.dumps({"DAMAGE_TAKEN_PENALTY": "0.3"}), "result": "regressed"}]
    new, reason = propose_from_memory(events, exps, {"DAMAGE_TAKEN_PENALTY": "0.1"})
    # must NOT re-propose the disproven knob -> falls through (low_hp high, so no DEATH branch)
    assert new is None


def test_propose_none_without_enough_deaths():
    new, reason = propose_from_memory(_deaths(3, health=10), [], {})
    assert new is None


def test_propose_high_hp_deaths_raise_death_penalty():
    events = _deaths(12, health=90)        # dying at high HP -> reckless fights
    new, reason = propose_from_memory(events, [], {"DEATH_PENALTY": "5.0"})
    assert new is not None
    assert float(new["DEATH_PENALTY"]) > 5.0


# --------------------------- adoption end-to-end (real SQLite) ---------------------------

def test_adopt_improved_experiments_writes_learned_config(tmp_path):
    from writer import db as _db
    con = _db.connect(str(tmp_path))
    con.execute(
        "INSERT INTO experiments (ts, hypothesis_id, param, old_val, new_val, result, "
        "confidence, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("2026-01-01", 3, json.dumps({"COVERAGE_REWARD": "2.5"}), "1.5", "2.5",
         "improved", 0.8, ""),
    )
    con.commit()
    con.close()
    adopted = adopt_improved_experiments(str(tmp_path))
    assert adopted.get("COVERAGE_REWARD") == "2.5"
    assert LearnedConfig(str(tmp_path)).values()["COVERAGE_REWARD"] == "2.5"
