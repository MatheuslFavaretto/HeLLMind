"""Tests for writer.frontier_store — Go-Explore frontier archive + goal sampling."""
import random

from writer.frontier_store import FrontierStore


def test_empty_archive_returns_no_goal(tmp_path):
    fs = FrontierStore(str(tmp_path))
    assert fs.sample_goal("MAP01", (0.0, 0.0)) is None


def test_merge_then_cells(tmp_path):
    fs = FrontierStore(str(tmp_path), cell_size=96.0)
    fs.merge("MAP01", [(1000.0, 0.0), (50.0, 0.0)])
    cells = fs.cells("MAP01")
    assert len(cells) == 2


def test_merge_buckets_nearby_positions(tmp_path):
    fs = FrontierStore(str(tmp_path), cell_size=96.0)
    # Two positions inside the same 96-unit cell collapse to one archived cell (visits=2).
    fs.merge("MAP01", [(10.0, 10.0), (20.0, 20.0)])
    cells = fs.cells("MAP01")
    assert len(cells) == 1
    assert cells[0][2] == 2  # visits


def test_sample_goal_prefers_far_cells(tmp_path):
    fs = FrontierStore(str(tmp_path), cell_size=96.0)
    fs.merge("MAP01", [(1500.0, 0.0)])     # far
    fs.merge("MAP01", [(100.0, 0.0)])      # within min_dist of spawn -> excluded
    g = fs.sample_goal("MAP01", (0.0, 0.0), rng=random.Random(1), min_dist=200.0)
    assert g == (1500.0, 0.0)


def test_sample_goal_excludes_too_close(tmp_path):
    fs = FrontierStore(str(tmp_path), cell_size=96.0)
    fs.merge("MAP01", [(50.0, 0.0)])  # only a near cell exists
    assert fs.sample_goal("MAP01", (0.0, 0.0), min_dist=200.0) is None


def test_persisted_across_instances(tmp_path):
    a = FrontierStore(str(tmp_path), cell_size=96.0)
    a.merge("MAP02", [(900.0, 0.0)])
    b = FrontierStore(str(tmp_path), cell_size=96.0)  # new process/env, same vault
    g = b.sample_goal("MAP02", (0.0, 0.0), rng=random.Random(0), min_dist=200.0)
    assert g == (900.0, 0.0)


def test_corrupt_file_safe(tmp_path):
    import os
    fs = FrontierStore(str(tmp_path))
    os.makedirs(fs.dir, exist_ok=True)
    with open(fs._path("MAP01"), "w") as f:
        f.write("not json")
    assert fs.cells("MAP01") == []      # never raises
    assert fs.sample_goal("MAP01", (0.0, 0.0)) is None
