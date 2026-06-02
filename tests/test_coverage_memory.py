"""Per-map persistent exploration heatmap: store, training-end callback, minimap overlay."""
import os

import cv2

from rl.coverage_callback import CoverageMemoryCallback
from writer.coverage_store import CoverageStore
from writer.minimap import render_minimap


# --------------------------- CoverageStore ---------------------------
def test_coverage_store_accumulates_across_runs(tmp_path):
    store = CoverageStore(str(tmp_path))
    assert store.load("MAP02") is None  # nothing yet

    store.merge("MAP02", {(0, 0): 2, (1, 0): 1}, walls=[[0, 0, 96, 0]])
    rec = store.merge("MAP02", {(1, 0): 3, (2, 2): 1})  # second run adds + overlaps

    assert rec["runs"] == 2
    assert rec["cells"]["1,0"] == 4          # 1 + 3 summed across runs
    assert rec["cells"]["0,0"] == 2
    assert rec["walls"] == [[0, 0, 96, 0]]   # geometry retained
    cells = store.load_cells("MAP02")
    assert sorted(c[:2] for c in cells) == [[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]]


def test_coverage_store_maps_are_isolated(tmp_path):
    store = CoverageStore(str(tmp_path))
    store.merge("MAP01", {(0, 0): 1})
    store.merge("MAP02", {(5, 5): 1})
    assert store.load("MAP01")["cells"] == {"0,0": 1}
    assert store.load("MAP02")["cells"] == {"5,5": 1}


# --------------------------- callback (flush at training end) ---------------------------
def _ep_info(map_name, cells, walls=None):
    doom = {"levels": {}, "deltas": {}, "action": 0,
            "coverage_cells": len(cells), "visited_cells": cells}
    if walls:
        doom["walls"] = walls
    return {"map": map_name, "doom": doom, "episode": {"r": 1.0, "l": 50}}


def test_callback_persists_only_at_training_end(tmp_path):
    store = CoverageStore(str(tmp_path))
    cb = CoverageMemoryCallback(store)

    # mid-episode step (no "episode" key) carries walls but no visited grid yet
    cb.locals = {"infos": [{"map": "MAP02",
                            "doom": {"walls": [[0, 0, 96, 0]], "levels": {}}}]}
    cb._on_step()
    cb.locals = {"infos": [_ep_info("MAP02", [[0, 0], [1, 1]])]}
    cb._on_step()
    cb.locals = {"infos": [_ep_info("MAP02", [[1, 1], [2, 2]])]}
    cb._on_step()

    assert store.load("MAP02") is None  # nothing written until training ends
    cb.on_training_end()
    rec = store.load("MAP02")
    assert rec["runs"] == 1
    assert rec["cells"]["1,1"] == 2  # visited in both episodes
    assert rec["walls"] == [[0, 0, 96, 0]]


# --------------------------- minimap overlay ---------------------------
def test_minimap_renders_memory_layer_only(tmp_path):
    out = os.path.join(tmp_path, "mem.png")
    mem = [[0, 0, 3], [1, 0, 1], [1, 1, 5], [2, 2, 2]]
    assert render_minimap([], out, memory_cells=mem) is True
    assert cv2.imread(out) is not None
