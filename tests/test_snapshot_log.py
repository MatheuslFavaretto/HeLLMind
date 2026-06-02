"""SnapshotLog: round-trip, numpy-type sanitization, and meta sidecar."""
import os

import numpy as np

from writer.snapshot_log import (
    SnapshotLog,
    _sanitize,
    log_path_for,
    meta_path_for,
    read_meta,
    write_meta,
)


def test_sanitize_numpy_types():
    out = _sanitize({"f": np.float32(1.5), "i": np.int64(3),
                     "arr": np.array([1, 2]), "nested": {"x": np.float64(2.0)}})
    assert out == {"f": 1.5, "i": 3, "arr": [1, 2], "nested": {"x": 2.0}}
    assert isinstance(out["f"], float) and isinstance(out["i"], int)


def test_append_and_read_all(tmp_path):
    p = os.path.join(tmp_path, "run.jsonl")
    log = SnapshotLog(p)
    log.append({"num_timesteps": 1000, "v": np.float32(0.5)})
    log.append({"num_timesteps": 2000, "v": 0.7})
    assert log.count == 2
    rows = SnapshotLog.read_all(p)
    assert [r["num_timesteps"] for r in rows] == [1000, 2000]
    assert rows[0]["v"] == 0.5  # numpy serializou ok


def test_init_truncates_previous_run(tmp_path):
    p = os.path.join(tmp_path, "run.jsonl")
    SnapshotLog(p).append({"a": 1})
    SnapshotLog(p)  # new run -> truncates the file
    assert SnapshotLog.read_all(p) == []


def test_read_all_missing_file(tmp_path):
    assert SnapshotLog.read_all(os.path.join(tmp_path, "nope.jsonl")) == []


def test_meta_roundtrip(tmp_path):
    p = meta_path_for(str(tmp_path), "run-x")
    write_meta(p, {"run_name": "run-x", "button_names": ["A", "B"]})
    assert read_meta(p)["button_names"] == ["A", "B"]
    assert read_meta(os.path.join(tmp_path, "missing.json")) is None


def test_path_helpers():
    assert log_path_for("/tmp/pending", "r1").endswith("r1.jsonl")
    assert meta_path_for("/tmp/pending", "r1").endswith("r1.meta.json")
