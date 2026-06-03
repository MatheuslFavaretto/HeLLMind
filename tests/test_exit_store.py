"""Tests for writer.exit_store — persisted per-map exit position (autonomy memory)."""
from writer.exit_store import ExitStore


def test_load_missing_is_none(tmp_path):
    s = ExitStore(str(tmp_path))
    assert s.load("MAP01") is None


def test_save_then_load_round_trip(tmp_path):
    s = ExitStore(str(tmp_path))
    s.save("MAP01", 123.5, 678.9)
    assert s.load("MAP01") == (123.5, 678.9)


def test_save_is_per_map(tmp_path):
    s = ExitStore(str(tmp_path))
    s.save("MAP01", 1.0, 2.0)
    s.save("MAP02", 3.0, 4.0)
    assert s.load("MAP01") == (1.0, 2.0)
    assert s.load("MAP02") == (3.0, 4.0)


def test_save_overwrites(tmp_path):
    s = ExitStore(str(tmp_path))
    s.save("MAP01", 1.0, 2.0)
    s.save("MAP01", 9.0, 9.0)  # exit is static, last writer wins
    assert s.load("MAP01") == (9.0, 9.0)


def test_shared_dir_visible_to_other_instance(tmp_path):
    """The whole point: one env writes, another (new instance, same vault) reads it."""
    writer = ExitStore(str(tmp_path))
    writer.save("MAP03", 50.0, 60.0)
    reader = ExitStore(str(tmp_path))  # simulates a different process/env
    assert reader.load("MAP03") == (50.0, 60.0)


def test_corrupt_file_returns_none(tmp_path):
    import os
    s = ExitStore(str(tmp_path))
    os.makedirs(s.dir, exist_ok=True)
    with open(s._path("MAP01"), "w") as f:
        f.write("{ not valid json")
    assert s.load("MAP01") is None  # never raises


def test_unsafe_map_name_sanitised(tmp_path):
    s = ExitStore(str(tmp_path))
    s.save("../evil/MAP", 1.0, 2.0)   # path traversal chars stripped
    assert s.load("../evil/MAP") == (1.0, 2.0)
