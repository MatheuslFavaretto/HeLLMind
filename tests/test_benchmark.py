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


def test_writers_produce_all_three_files(tmp_path):
    results = {"baseline": _aggregate([{k: 0.1 for k in METRIC_KEYS}]),
               "full": _aggregate([{k: 0.5 for k in METRIC_KEYS}])}
    payload = {"generated": "now", "map": "MAP01", "steps": 50000,
               "seeds": [42], "episodes": 20, "configs": results}
    _write_json(str(tmp_path), payload)
    _write_csv(str(tmp_path), results)
    _write_md(str(tmp_path), payload)
    for fname in ("benchmark.json", "benchmark.csv", "benchmark.md"):
        assert os.path.exists(tmp_path / fname)
    loaded = json.loads((tmp_path / "benchmark.json").read_text())
    assert "full" in loaded["configs"]
    assert "baseline" in (tmp_path / "benchmark.md").read_text()
    assert "config" in (tmp_path / "benchmark.csv").read_text()
