"""Tests for rl.eureka — pure pieces of the evolutionary reward search (no LLM/training)."""
import random

from rl.eureka import (
    EUREKA_BOUNDS,
    clamp_env,
    mutate_heuristic,
    propose_candidates,
    select_best,
)


def test_clamp_env_respects_bounds():
    env = {k: str(hi * 10) for k, (lo, hi) in EUREKA_BOUNDS.items()}  # all over ceiling
    clamped = clamp_env(env)
    for k, (lo, hi) in EUREKA_BOUNDS.items():
        assert lo <= float(clamped[k]) <= hi


def test_clamp_env_floor():
    env = {k: str(lo - 100) for k, (lo, hi) in EUREKA_BOUNDS.items()}  # under floor
    clamped = clamp_env(env)
    for k, (lo, hi) in EUREKA_BOUNDS.items():
        assert float(clamped[k]) >= lo


def test_clamp_ignores_non_numeric():
    env = {"HIT_REWARD": "not_a_number", "MAPS": "MAP01"}
    out = clamp_env(env)
    assert out["MAPS"] == "MAP01"  # untouched, no crash


def test_mutate_stays_within_bounds():
    base = {k: str((lo + hi) / 2) for k, (lo, hi) in EUREKA_BOUNDS.items()}
    rng = random.Random(0)
    for _ in range(50):
        m = mutate_heuristic(base, rng)
        for k, (lo, hi) in EUREKA_BOUNDS.items():
            assert lo <= float(m[k]) <= hi


def test_mutate_actually_changes_something():
    base = {k: str((lo + hi) / 2) for k, (lo, hi) in EUREKA_BOUNDS.items()}
    m = mutate_heuristic(base, random.Random(1))
    assert any(m[k] != base[k] for k in EUREKA_BOUNDS)  # not a no-op


def test_select_best_picks_max_score():
    cands = [
        {"env": {"A": "1"}, "score": 0.5},
        {"env": {"A": "2"}, "score": 1.5},
        {"env": {"A": "3"}, "score": 0.9},
    ]
    assert select_best(cands)["score"] == 1.5


def test_select_best_ignores_unscored():
    cands = [{"env": {}, "score": None}, {"env": {}, "score": 0.2}]
    assert select_best(cands)["score"] == 0.2


def test_select_best_all_none_returns_none():
    assert select_best([{"env": {}, "score": None}]) is None


def test_propose_candidates_falls_back_without_ollama(monkeypatch):
    # Force the LLM path to fail -> heuristic mutation must produce `n` clamped candidates.
    import rl.eureka as eu

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no ollama")

    monkeypatch.setattr("ollama.Client", Boom, raising=False)
    from types import SimpleNamespace
    cfg = SimpleNamespace(ollama_host="x", llm_model="m", llm_num_ctx=2048,
                          llm_keep_alive="5m")
    base = {k: str((lo + hi) / 2) for k, (lo, hi) in EUREKA_BOUNDS.items()}
    cands = propose_candidates(cfg, base, {}, [], n=4, rng=random.Random(0))
    assert len(cands) == 4
    for c in cands:
        for k, (lo, hi) in EUREKA_BOUNDS.items():
            assert lo <= float(c[k]) <= hi
