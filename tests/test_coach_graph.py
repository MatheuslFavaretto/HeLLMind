"""Tests for the LangGraph coach (rl.coach_graph) — V2 Phase 4.

Covers the graph's node logic and the algo-aware un-freeze knob (PPO→ENT_COEF,
DQN→DQN_EPS_FINAL) so the DQN auto-loop doesn't tune a no-op knob via the graph path.
"""
import pytest

pytest.importorskip("langgraph")

from rl.coach_graph import (
    CoachGraph, node_diagnose, node_hypothesize, _score,
)


def _state(**overrides):
    base = {
        "metrics": {}, "env": {}, "history": [], "use_llm": False,
        "memory_dir": "/tmp/hellmind-test", "algo": "ppo",
        "diagnosis": "", "hypothesis": "", "next_env": {},
        "reason": "", "score": 0.0, "kept": True, "log": [],
    }
    base.update(overrides)
    return base


def test_score_matches_autonomous_formula():
    from rl.autonomous import score
    m = {"exit_rate": 0.0, "exit_progress": 0.2, "explored_fraction": 0.1,
         "shooting_accuracy": 0.05, "kills_per_episode": 1.0}
    assert _score(m) == pytest.approx(score(m))


def test_diagnose_passive_overall():
    m = {"kills_per_episode": 0.2, "timeout_rate": 0.1, "explored_fraction": 0.3,
         "death_rate": 0.1}
    out = node_diagnose(_state(metrics=m))
    assert out["diagnosis"] == "passive_overall"


def test_diagnose_stuck_at_spawn():
    m = {"explored_fraction": 0.05, "timeout_rate": 0.2, "death_rate": 0.1,
         "kills_per_episode": 1.0}
    out = node_diagnose(_state(metrics=m))
    assert out["diagnosis"] == "stuck_at_spawn"


def test_hypothesize_uses_ent_coef_for_ppo():
    out = node_hypothesize(_state(diagnosis="passive_overall", algo="ppo"))
    assert "ENT_COEF" in out["hypothesis"]
    assert "DQN_EPS_FINAL" not in out["hypothesis"]


def test_hypothesize_uses_epsilon_for_dqn():
    # The DQN path must NOT mention ENT_COEF (a no-op for QR-DQN).
    out = node_hypothesize(_state(diagnosis="passive_overall", algo="dqn"))
    assert "DQN_EPS_FINAL" in out["hypothesis"]
    assert "ENT_COEF" not in out["hypothesis"]


def test_graph_run_dqn_does_not_touch_ent_coef():
    # End-to-end: a passive-combat DQN iteration should raise DQN_EPS_FINAL, never ENT_COEF.
    class _Cfg:
        memory_dir = "/tmp/hellmind-test-coach"
    coach = CoachGraph(_Cfg(), use_llm=False, algo="dqn")
    env = {"ENT_COEF": "0.03", "ENGAGEMENT_REWARD": "0.01", "EPISODE_TIMEOUT": "2100"}
    m = {"kills_per_episode": 0.2, "timeout_rate": 0.1, "explored_fraction": 0.3,
         "death_rate": 0.1, "exit_rate": 0.0, "shooting_accuracy": 0.2}
    result = coach.run(metrics=m, env=env, history=[])
    nxt = result["next_env"]
    assert float(nxt.get("ENT_COEF", 0.03)) == 0.03      # untouched
    assert "DQN_EPS_FINAL" in nxt                          # the real DQN lever moved
