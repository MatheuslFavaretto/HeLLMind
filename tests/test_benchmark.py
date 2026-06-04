"""Tests for the ablation benchmark's pure pieces (aggregation + report writers).
The training/eval orchestration needs ViZDoom, so it's exercised manually, not here."""
import json
import os

import pytest

from rl.benchmark import CONFIGS, METRIC_KEYS, _aggregate, _write_csv, _write_json, _write_md


def test_configs_are_cumulative_layers():
    # Each layer should be a superset-ish of the previous capability set.
    assert CONFIGS["baseline"] == {}
    assert CONFIGS["rnd"].get("USE_RND") == "1"
    assert CONFIGS["memory"].get("MEMORY_ENABLED") == "1"
    assert CONFIGS["full"].get("COMBAT_EXPLORE_SPLIT") == "1"


def test_aggregate_mean_and_std():
    per_seed = [
        {"exit_rate": 0.0, "explored_fraction": 0.10, "kills_per_episode": 1.0,
         "death_rate": 0.2, "combat_engagement": 0.3, "mean_base_reward": 5.0},
        {"exit_rate": 0.0, "explored_fraction": 0.20, "kills_per_episode": 3.0,
         "death_rate": 0.4, "combat_engagement": 0.5, "mean_base_reward": 7.0},
    ]
    agg = _aggregate(per_seed)
    assert agg["explored_fraction"]["mean"] == pytest.approx(0.15)
    assert agg["kills_per_episode"]["mean"] == pytest.approx(2.0)
    assert agg["kills_per_episode"]["std"] > 0
    assert agg["_seeds"] == 2


def test_config_score_ranks_better_config_higher():
    from rl.benchmark import _config_score
    weak = _aggregate([{"explored_fraction": 0.05}])
    strong = _aggregate([{"exit_rate": 0.5, "exit_progress": 0.8, "explored_fraction": 0.4}])
    assert _config_score(strong) > _config_score(weak)


def test_writers_produce_all_four_files(tmp_path):
    from rl.benchmark import _config_score, _write_html
    results = {}
    for name, ev in (("baseline", 0.1), ("full", 0.5)):
        agg = _aggregate([{k: ev for k in METRIC_KEYS}])
        agg["_score"] = _config_score(agg)
        results[name] = agg
    best = max(results, key=lambda n: results[n]["_score"])
    payload = {"generated": "now", "map": "MAP01", "steps": 50000,
               "seeds": [42], "episodes": 20, "best": best, "configs": results}
    _write_json(str(tmp_path), payload)
    _write_csv(str(tmp_path), results)
    _write_md(str(tmp_path), payload)
    _write_html(str(tmp_path), payload)
    for fname in ("benchmark.json", "benchmark.csv", "benchmark.md", "benchmark.html"):
        assert os.path.exists(tmp_path / fname)
    html = (tmp_path / "benchmark.html").read_text()
    assert "<table" in html and "score" in html
    assert "best" in (tmp_path / "benchmark.md").read_text().lower()


def test_fmt_durations():
    from rl.benchmark import _fmt
    assert _fmt(45) == "45s"
    assert _fmt(75) == "1m15s"
    assert _fmt(3700) == "1h01m"
    assert _fmt(-5) == "0s"
