"""Exploration + completion + autonomy features."""
import types

import numpy as np

from rl.autonomous import BOUNDS, llm_propose, propose, propose_next, score
from config import Config
from rl.campaign_callbacks import (
    combined_map_weights,
    frontier_step_weights,
    map_step_weights,
)
from instrumentation.stats_tracker import StatsTracker


# --------------------------- frontier curriculum ---------------------------
def test_frontier_weights_favor_underexplored_map():
    maps = ["MAP01", "MAP02"]
    events = [
        {"map": "MAP01", "coverage": 5, "type": "timeout"},   # barely explored
        {"map": "MAP02", "coverage": 80, "type": "timeout"},  # well explored
    ]
    w = frontier_step_weights(events, maps)
    assert w["MAP01"] > w["MAP02"]               # under-explored gets more budget
    assert abs((w["MAP01"] + w["MAP02"]) / 2 - 1.0) < 1e-6  # normalized mean 1.0


def test_combined_weights_normalized_and_no_memory_is_uniform():
    maps = ["MAP01", "MAP02", "MAP03"]
    assert combined_map_weights([], maps) == {m: 1.0 for m in maps}
    # All maps seen; MAP02 is both the deadliest AND the least explored -> most focus.
    events = [
        {"map": "MAP01", "type": "timeout", "coverage": 50},
        {"map": "MAP02", "type": "death", "coverage": 5},
        {"map": "MAP03", "type": "timeout", "coverage": 40},
    ]
    w = combined_map_weights(events, maps)
    assert abs(sum(w.values()) / len(maps) - 1.0) < 1e-6
    assert w["MAP02"] > w["MAP01"] and w["MAP02"] > w["MAP03"]


# --------------------------- autonomy supervisor ---------------------------
def test_score_rewards_the_goal():
    base = {"exit_rate": 0.0, "explored_fraction": 0.2,
            "kills_per_episode": 1.0, "shooting_accuracy": 0.02}
    better_exit = {**base, "exit_rate": 0.5}
    better_explore = {**base, "explored_fraction": 0.8}
    assert score(better_exit) > score(base)
    assert score(better_explore) > score(base)


def test_propose_stays_within_bounds_and_targets_weakness():
    # Start near every ceiling so a bump would overflow -> must be clamped.
    env = {k: str(hi) for k, (lo, hi) in BOUNDS.items()}
    new, reason = propose(env, {"explored_fraction": 0.1, "exit_rate": 0.0,
                                "kills_per_episode": 0.0, "shooting_accuracy": 0.0})
    for k, (lo, hi) in BOUNDS.items():
        assert lo <= float(new[k]) <= hi          # guardrails hold (clamped)
    assert "COVERAGE_REWARD" in reason             # targeted the weakest (exploration)


# --------------------------- LLM-driven proposer ---------------------------
def _fake_llm(monkeypatch, tweaks, summary="combat looks off"):
    """Wire writer.suggest's pieces so llm_propose runs without Ollama/memory."""
    monkeypatch.setattr("writer.memory_store.MemoryStore.read_events",
                        staticmethod(lambda _d: [{"x": 1}]))
    monkeypatch.setattr("writer.reflect.aggregate_events", lambda _e: {"total": 50})

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

        def generate_reward_suggestions(self, stats, weights):
            tw = [types.SimpleNamespace(knob=k, suggested=v, reason="r") for k, v in tweaks]
            return types.SimpleNamespace(summary=summary, tweaks=tw)

    monkeypatch.setattr("writer.llm_client.LLMWriter", FakeLLM)


def test_llm_propose_maps_and_clamps_combat_knobs(monkeypatch):
    # LLM asks for a huge death penalty + a hit reward; both must clamp to BOUNDS.
    _fake_llm(monkeypatch, [("death_penalty", 999.0), ("hit_reward", 3.0)])
    env = {"DEATH_PENALTY": "5.0", "HIT_REWARD": "2.0"}
    new, reason = llm_propose(Config(), env, {"shooting_accuracy": 0.05})
    assert float(new["DEATH_PENALTY"]) == BOUNDS["DEATH_PENALTY"][1]  # clamped to ceiling
    assert float(new["HIT_REWARD"]) == 3.0
    assert reason.startswith("LLM:")


def test_llm_propose_falls_back_when_unavailable(monkeypatch):
    # Too few events -> no LLM suggestion -> None (caller keeps the heuristic).
    monkeypatch.setattr("writer.memory_store.MemoryStore.read_events",
                        staticmethod(lambda _d: []))
    monkeypatch.setattr("writer.reflect.aggregate_events", lambda _e: {"total": 0})
    assert llm_propose(Config(), {"HIT_REWARD": "2.0"}, {"shooting_accuracy": 0.0}) is None


def test_subprocess_env_coerces_floats_to_str():
    # propose() puts float reward weights into the env; subprocess needs str values, so a
    # resume after a tweak must not crash with "expected str ... not float".
    from rl.autonomous import _subprocess_env
    env = {"COVERAGE_REWARD": 0.75, "N_ENVS": 4, "MAPS": "MAP02"}
    out = _subprocess_env(env)
    assert out["COVERAGE_REWARD"] == "0.75" and out["N_ENVS"] == "4"
    assert all(isinstance(v, str) for v in out.values())  # every value subprocess-safe


def test_propose_next_heuristic_only_matches_propose():
    env = {k: str(hi) for k, (lo, hi) in BOUNDS.items()}
    m = {"explored_fraction": 0.1, "exit_rate": 0.0,
         "kills_per_episode": 0.0, "shooting_accuracy": 0.0}
    assert propose_next(Config(), dict(env), m, use_llm=False) == propose(dict(env), m)


def test_propose_next_layers_llm_on_top_of_heuristic(monkeypatch):
    _fake_llm(monkeypatch, [("hit_reward", 4.0)])
    # Healthy combat/exploration so the heuristic anneals COVERAGE; LLM then bumps HIT.
    env = {k: str((lo + hi) / 2) for k, (lo, hi) in BOUNDS.items()}
    m = {"explored_fraction": 0.9, "exit_rate": 0.5,
         "kills_per_episode": 2.0, "shooting_accuracy": 0.3}
    new, reason = propose_next(Config(), env, m, use_llm=True)
    assert float(new["HIT_REWARD"]) == 4.0          # LLM combat tweak applied
    assert "LLM:" in reason and "COVERAGE_REWARD" in reason  # both layers present


# --------------------------- stats: exit-rate + coverage ---------------------------
def _ep_info(terminal, cells, walls=None):
    doom = {
        "deltas": {"killcount": 0.0, "hitcount": 0.0, "distance": 0.0,
                   "damage_taken": 0.0},
        "levels": {"health": 50.0, "ammo2": 10.0, "position_x": 0.0,
                   "position_y": 0.0, "selected_weapon": 2.0},
        "action": 0, "success": terminal == "exit", "terminal": terminal,
        "coverage_cells": cells, "base_return": 1.0,
    }
    if walls:
        doom["walls"] = walls
    info = {"doom": doom, "map": "MAP02", "episode": {"r": 1.0, "l": 100}}
    return info


def test_exit_rate_and_true_coverage():
    t = StatsTracker(button_names=["TURN_LEFT", "TURN_RIGHT", "ATTACK"])
    walls = [[0.0, 0.0, 960.0, 0.0], [0.0, 0.0, 0.0, 960.0]]  # 10x10 cells @96
    # 3 episodes: one reaches the exit, two time out.
    for term, cells in [("exit", 30), ("timeout", 10), ("timeout", 12)]:
        t.update([_ep_info(term, cells, walls)], np.array([0]))
    snap = t.snapshot(1000)
    assert abs(snap["exit_rate"] - 1 / 3) < 1e-9
    assert snap["terminals"]["exit"] == 1
    cov = snap["map_coverage"]
    assert cov["source"] == "walls"
    assert 0.0 < cov["explored_fraction"] <= 1.0
