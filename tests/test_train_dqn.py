"""Tests for the QR-DQN training engine (V2 Phase 1)."""
import os
import tempfile

import pytest

from rl.train_dqn import _best_device, _dqn_prefix, _latest_dqn_checkpoint


def test_best_device_returns_valid_string():
    d = _best_device()
    assert d in ("cuda", "mps", "cpu")


def test_dqn_prefix_encodes_gamevars():
    assert "gv" in _dqn_prefix(11, True)
    assert "gv" not in _dqn_prefix(11, False)
    assert "a11" in _dqn_prefix(11, False)
    assert "qrdqn" in _dqn_prefix(15, True)


def test_latest_dqn_checkpoint_returns_none_when_empty(tmp_path):
    assert _latest_dqn_checkpoint(str(tmp_path), "qrdqn_test") is None


def test_latest_dqn_checkpoint_picks_newest(tmp_path):
    import time
    p1 = tmp_path / "qrdqn_test_1000.zip"; p1.write_text("a")
    time.sleep(0.05)
    p2 = tmp_path / "qrdqn_test_5000.zip"; p2.write_text("b")
    assert _latest_dqn_checkpoint(str(tmp_path), "qrdqn_test") == str(p2)


@pytest.mark.slow
def test_qrdqn_smoke_train(tmp_path):
    """Full smoke: creates a QR-DQN model on the real env and trains 100 steps."""
    from config import Config
    from rl.train_dqn import train

    os.environ.update({"DQN_BUFFER": "300", "DQN_WARMUP": "50", "DQN_BATCH": "16"})
    cfg = Config()
    cfg.n_envs = 1
    cfg.docs_enabled = False
    cfg.memory_enabled = False
    cfg.checkpoint_dir = str(tmp_path)
    cfg.tensorboard_log = None

    path = train(cfg, cfg.maps[0], timesteps=100, fresh=True, n_envs=1, verbose=0)
    assert os.path.exists(path)
