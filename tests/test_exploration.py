"""Exploration + completion + autonomy features."""
import types

import numpy as np

from rl.autonomous import BOUNDS, llm_propose, propose, propose_next, score
from config import Config
from rl.campaign_callbacks import combined_map_weights, frontier_step_weights
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
    new, reason = propose(env, {"explored_fraction": 0.05, "exit_rate": 0.0,
                                "kills_per_episode": 0.0, "shooting_accuracy": 0.0})
    for k, (lo, hi) in BOUNDS.items():
        assert lo <= float(new[k]) <= hi          # guardrails hold (clamped)
    assert "COVERAGE_REWARD" in reason             # severe under-exploration targeted first


def test_propose_targets_death_rate():
    # Dying a lot is the root cause -> raise damage/death penalties before anything else.
    env = {"DAMAGE_TAKEN_PENALTY": "0.2", "DEATH_PENALTY": "4.0", "EPISODE_TIMEOUT": "2100"}
    new, reason = propose(env, {"explored_fraction": 0.3, "death_rate": 0.8,
                                "kills_per_episode": 1.0, "timeout_rate": 0.1})
    assert float(new["DAMAGE_TAKEN_PENALTY"]) > 0.2
    assert "death_rate" in reason


def test_propose_targets_passivity_with_entropy():
    # Barely kills but not timing out -> argmax collapse -> raise ENT_COEF + ENGAGEMENT.
    env = {"ENT_COEF": "0.03", "ENGAGEMENT_REWARD": "0.01", "EPISODE_TIMEOUT": "2100"}
    new, reason = propose(env, {"explored_fraction": 0.3, "death_rate": 0.1,
                                "kills_per_episode": 0.2, "timeout_rate": 0.1})
    assert float(new["ENT_COEF"]) > 0.03
    assert "ENT_COEF" in reason


def test_score_caps_kills_so_camper_loses_to_explorer():
    # Regression: kills/ep is unbounded, so an un-normalised 0.5*kills once let a
    # spawn-camper (many kills, no exploration) outscore a real explorer.
    camper = {"exit_rate": 0.0, "explored_fraction": 0.03,
              "kills_per_episode": 4.0, "shooting_accuracy": 0.0}
    explorer = {"exit_rate": 0.0, "explored_fraction": 0.40,
                "kills_per_episode": 0.5, "shooting_accuracy": 0.0}
    assert score(explorer) > score(camper)


def test_score_kills_contribution_is_capped():
    # 5 vs 50 kills must score the same (cap at 5) — kills is a tiebreaker, not the goal.
    base = {"exit_rate": 0.0, "explored_fraction": 0.0, "shooting_accuracy": 0.0}
    assert score({**base, "kills_per_episode": 5.0}) == score({**base, "kills_per_episode": 50.0})


def test_propose_extends_timeout_when_episodes_time_out():
    # > 80% timeouts + low exploration => the episode is too short, raise EPISODE_TIMEOUT.
    env = {"EPISODE_TIMEOUT": "2100", "COVERAGE_REWARD": "1.5", "FRONTIER_REWARD": "0.05"}
    new, reason = propose(env, {"timeout_rate": 0.9, "explored_fraction": 0.05,
                                "exit_rate": 0.0, "kills_per_episode": 1.0,
                                "shooting_accuracy": 0.0})
    assert float(new["EPISODE_TIMEOUT"]) > 2100.0
    assert "EPISODE_TIMEOUT" in reason


def test_propose_no_timeout_extension_when_exploring():
    # Timeouts but decent exploration => DON'T extend; the agent is using its time.
    env = {"EPISODE_TIMEOUT": "2100", "COVERAGE_REWARD": "1.5", "FRONTIER_REWARD": "0.05"}
    new, reason = propose(env, {"timeout_rate": 0.9, "explored_fraction": 0.40,
                                "exit_rate": 0.0, "kills_per_episode": 1.0,
                                "shooting_accuracy": 0.0})
    assert float(new["EPISODE_TIMEOUT"]) == 2100.0


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


def test_propose_next_heuristic_only_matches_propose(tmp_path):
    # With no LLM and an EMPTY memory (no death history), propose_next is the pure heuristic.
    cfg = Config()
    cfg.memory_dir = str(tmp_path)   # isolate: no events -> memory policy stays silent
    env = {k: str(hi) for k, (lo, hi) in BOUNDS.items()}
    m = {"explored_fraction": 0.1, "exit_rate": 0.0,
         "kills_per_episode": 0.0, "shooting_accuracy": 0.0}
    assert propose_next(cfg, dict(env), m, use_llm=False) == propose(dict(env), m)


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


def test_write_log_handles_non_contiguous_iters(tmp_path):
    # Regression: failed iterations are skipped, so history iter numbers have GAPS (0,2,4).
    # write_log must index by list position, not history[iter-1], or it IndexErrors.
    import os
    from rl.autonomous import write_log
    cfg = Config()
    cfg.memory_dir = str(tmp_path)
    cfg.vault_path = str(tmp_path)
    os.makedirs(os.path.join(str(tmp_path), cfg.dir_index), exist_ok=True)
    base_m = {"explored_fraction": 0.05, "exit_rate": 0.0,
              "kills_per_episode": 1.5, "shooting_accuracy": 0.0}
    hist = [
        {"iter": 0, "score": 0.30, "kept": True, "reason": "baseline",
         "env": {"COVERAGE_REWARD": "2.0"}, "metrics": base_m},
        {"iter": 2, "score": 0.30, "kept": True, "reason": "adjust",
         "env": {"COVERAGE_REWARD": "2.4"}, "metrics": base_m},
        {"iter": 4, "score": 0.46, "kept": True, "reason": "adjust",
         "env": {"COVERAGE_REWARD": "2.8"}, "metrics": {**base_m, "explored_fraction": 0.06}},
    ]
    write_log(cfg, hist)  # must not raise IndexError
    assert os.path.exists(os.path.join(str(tmp_path), cfg.dir_index, "Autonomy Log.md"))
