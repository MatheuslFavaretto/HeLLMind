"""Tests for instrumentation.prometheus_exporter.flatten_metrics (pure; no prometheus dep)."""
from instrumentation.prometheus_exporter import flatten_metrics


def test_flattens_scalars_and_dicts():
    m = {"explored_fraction": 0.25, "aim_offset": 0.93,
         "reward_breakdown": {"combat": 0.1, "explore": 0.82},
         "weapons_used": {"slot_2": 0.7}}
    triples = flatten_metrics(m)
    names = {(n, tuple(sorted(l.items()))) for n, l, _ in triples}
    assert ("hellmind_explored_fraction", ()) in names
    assert ("hellmind_reward_breakdown", (("item", "combat"),)) in names
    assert ("hellmind_reward_breakdown", (("item", "explore"),)) in names
    assert ("hellmind_weapons_used", (("item", "slot_2"),)) in names
    vals = {n: v for n, l, v in triples if not l}
    assert vals["hellmind_aim_offset"] == 0.93


def test_skips_arrays_and_nonnumeric():
    m = {"path_cells": [[1, 2, 3]], "map_walls": [[0, 0, 1, 1]], "map": "MAP01",
         "kills_per_episode": 7.0}
    triples = flatten_metrics(m)
    names = [n for n, _, _ in triples]
    assert "hellmind_kills_per_episode" in names
    assert all("path_cells" not in n and "map_walls" not in n for n in names)
    assert "hellmind_map" not in names            # string skipped


def test_empty_is_safe():
    assert flatten_metrics({}) == []
    assert flatten_metrics(None) == []
