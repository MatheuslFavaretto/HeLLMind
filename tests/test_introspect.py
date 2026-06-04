"""Tests for rl.introspect — the file-based intelligence stats (no ViZDoom/torch needed)."""
import json
import os

from rl.introspect import best_run, cognition_stats, disk_usage, training_stats


def test_brain_report_missing_path():
    from rl.introspect import brain_report
    assert brain_report("/nope/nope.zip") == {"exists": False}
    assert brain_report("") == {"exists": False}


def test_training_stats_counts_steps_and_checkpoints(tmp_path):
    ck = str(tmp_path)
    for n in (50000, 100000, 150000):
        open(os.path.join(ck, f"ppo_campaign_a11_sp_{n}_steps.zip"), "w").close()
    open(os.path.join(ck, "ppo_campaign_a11_sp_final.zip"), "w").close()
    # a different family must NOT be counted
    open(os.path.join(ck, "ppo_campaign_a11_250000_steps.zip"), "w").close()
    st = training_stats(ck, "ppo_campaign_a11_sp")
    assert st["total_steps"] == 150000
    assert st["checkpoints"] == 4   # 3 steps + 1 final


def test_training_stats_empty(tmp_path):
    st = training_stats(str(tmp_path), "ppo_campaign_a11_sp")
    assert st == {"total_steps": 0, "checkpoints": 0}


def test_best_run_picks_highest_score(tmp_path):
    p = os.path.join(str(tmp_path), "autonomy.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"iter": 0, "score": 0.5, "metrics": {"exit_rate": 0.0}}) + "\n")
        f.write(json.dumps({"iter": 1, "score": 1.8, "metrics": {"exit_rate": 0.2}}) + "\n")
        f.write(json.dumps({"iter": 2, "score": 0.9}) + "\n")
    br = best_run(str(tmp_path))
    assert br["score"] == 1.8 and br["iter"] == 1
    assert br["source"] == "autonomy.jsonl"


def test_best_run_none_when_empty(tmp_path):
    assert best_run(str(tmp_path)) is None


def test_cognition_stats_counts_events_and_frontier(tmp_path):
    mem = str(tmp_path)
    os.makedirs(os.path.join(mem, "episodic"))
    with open(os.path.join(mem, "episodic", "events.jsonl"), "w") as f:
        f.write('{"type":"death"}\n{"type":"exit"}\n{"type":"timeout"}\n')
    os.makedirs(os.path.join(mem, "frontier"))
    with open(os.path.join(mem, "frontier", "MAP01.json"), "w") as f:
        json.dump({"cells": {"0,0": {}, "1,0": {}}}, f)
    os.makedirs(os.path.join(mem, "exits"))
    open(os.path.join(mem, "exits", "MAP01.json"), "w").close()
    cg = cognition_stats(mem)
    assert cg["events"] == 3
    assert cg["frontier_cells"] == 2
    assert cg["exits_known"] == 1


def test_cognition_stats_empty(tmp_path):
    cg = cognition_stats(str(tmp_path))
    assert cg["events"] == 0 and cg["frontier_cells"] == 0


def test_disk_usage_reports_sizes(tmp_path):
    from types import SimpleNamespace
    ck = tmp_path / "ck"; ck.mkdir()
    (ck / "brain.zip").write_bytes(b"x" * 2000)
    mem = tmp_path / "mem"; mem.mkdir()
    (mem / "events.jsonl").write_bytes(b"y" * 1000)
    cfg = SimpleNamespace(checkpoint_dir=str(ck), memory_dir=str(mem), vault_path=str(tmp_path))
    du = disk_usage(cfg)
    assert du["checkpoints_mb"] >= 0.0
    assert du["memory_mb"] >= 0.0
    assert du["vault_total_mb"] >= du["checkpoints_mb"]  # vault contains everything
