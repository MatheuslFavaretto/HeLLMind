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


def test_merge_stamps_generation(tmp_path):
    fs = FrontierStore(str(tmp_path), cell_size=96.0)
    fs.merge("MAP01", [(1000.0, 0.0)])
    fs.merge("MAP01", [(1000.0, 0.0)])
    rec = fs.load("MAP01")
    assert rec["gen"] == 2
    assert rec["cells"]["10,0"]["last_gen"] == 2  # re-stamped on the 2nd merge


def test_aging_prunes_stale_cells(tmp_path):
    fs = FrontierStore(str(tmp_path), cell_size=96.0)
    fs.merge("MAP01", [(5000.0, 0.0)])              # cell A, gen 1
    for _ in range(30):                              # 30 merges touching only cell B
        fs.merge("MAP01", [(50.0, 0.0)])
    keys = {f"{int(x/96)},{int(y/96)}" for (x, y, _) in fs.cells("MAP01")}
    assert "52,0" not in keys                        # cell A aged out (max_age=25)


def test_edge_cell_preferred_over_interior(tmp_path):
    # An isolated far cell (an edge/boundary) should be sampled far more than a far cell
    # buried inside a dense, well-trodden blob of the same distance.
    fs = FrontierStore(str(tmp_path), cell_size=96.0)
    far = 3000.0
    # interior far cell surrounded by neighbours, all heavily visited
    blob = [(far, 0.0)]
    for dx in (-96, 0, 96):
        for dy in (-96, 0, 96):
            blob += [(far + dx, dy)] * 5
    fs.merge("MAP01", blob)
    # isolated edge cell, same distance, on the other side
    fs.merge("MAP01", [(-far, 0.0)])
    rng = random.Random(0)
    picks = [fs.sample_goal("MAP01", (0.0, 0.0), rng=rng) for _ in range(200)]
    edge = sum(1 for p in picks if p and p[0] < 0)
    assert edge > len(picks) * 0.5  # the lonely edge wins despite equal distance


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
