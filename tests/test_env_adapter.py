"""Tests for doom.env_adapter — Phase 6 generalisation interface."""
import numpy as np
import pytest

from doom.env_adapter import GymAdapter, EnvAdapter, make_adapter, register_adapter


# ---------------------------------------------------------------------------
# Minimal fake gymnasium env
# ---------------------------------------------------------------------------

class _FakeObsSpace:
    n = 16
    shape = (4,)
    dtype = np.float32
    def sample(self):
        return np.zeros(4, dtype=np.float32)


class _FakeActionSpace:
    n = 2
    def sample(self):
        return 0


class _FakeGymEnv:
    observation_space = _FakeObsSpace()
    action_space = _FakeActionSpace()
    spec = None

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        obs = np.ones(4, dtype=np.float32) * action
        done = (action == 1)  # action 1 terminates
        return obs, 1.0, done, False, {}

    def close(self):
        pass


# ---------------------------------------------------------------------------
# GymAdapter
# ---------------------------------------------------------------------------

def test_gym_adapter_is_env_adapter():
    adapter = GymAdapter(_FakeGymEnv(), env_name="fake")
    assert isinstance(adapter, EnvAdapter)


def test_gym_adapter_reset():
    adapter = GymAdapter(_FakeGymEnv())
    obs, info = adapter.reset()
    assert obs.shape == (4,)


def test_gym_adapter_step():
    adapter = GymAdapter(_FakeGymEnv())
    adapter.reset()
    obs, reward, term, trunc, info = adapter.step(0)
    assert reward == 1.0
    assert not term


def test_gym_adapter_button_names():
    adapter = GymAdapter(_FakeGymEnv())
    names = adapter.button_names()
    assert len(names) == 2
    assert names[0] == "action_0"


def test_gym_adapter_telemetry_timeout():
    adapter = GymAdapter(_FakeGymEnv(), env_name="cartpole")
    adapter.reset()
    adapter.step(0)
    t = adapter.telemetry()
    assert t["map"] == "cartpole"
    assert t["type"] in ("timeout", "success")
    assert t["kills"] == 0
    assert 0.0 <= t["coverage"] <= 1.0
    assert t["length"] == 1


def test_gym_adapter_telemetry_on_termination():
    adapter = GymAdapter(_FakeGymEnv())
    adapter.reset()
    adapter.step(1)  # action 1 terminates in _FakeGymEnv
    t = adapter.telemetry()
    assert t["type"] == "success"


def test_gym_adapter_coverage_increases():
    adapter = GymAdapter(_FakeGymEnv())
    adapter.reset()
    for action in range(2):
        adapter.step(action)
    t = adapter.telemetry()
    assert t["coverage"] > 0.0


def test_gym_adapter_episode_length():
    adapter = GymAdapter(_FakeGymEnv())
    adapter.reset()
    for _ in range(5):
        adapter.step(0)
    assert adapter.telemetry()["length"] == 5


def test_gym_adapter_reset_clears_state():
    adapter = GymAdapter(_FakeGymEnv())
    adapter.reset()
    for _ in range(10):
        adapter.step(0)
    adapter.reset()
    assert adapter._episode_length == 0
    assert adapter._total_reward == 0.0


# ---------------------------------------------------------------------------
# make_adapter + registry
# ---------------------------------------------------------------------------

def test_make_adapter_gym():
    adapter = make_adapter("gym", _FakeGymEnv(), env_name="test")
    assert isinstance(adapter, GymAdapter)


def test_make_adapter_unknown():
    with pytest.raises(ValueError, match="Unknown adapter"):
        make_adapter("nonexistent_env")


def test_register_adapter():
    class MyAdapter(GymAdapter):
        pass
    register_adapter("my_custom", MyAdapter)
    adapter = make_adapter("my_custom", _FakeGymEnv())
    assert isinstance(adapter, MyAdapter)


# ---------------------------------------------------------------------------
# Telemetry contract — required keys present
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("type", "map", "kills", "coverage", "health", "length")

def test_telemetry_has_required_keys():
    adapter = GymAdapter(_FakeGymEnv())
    adapter.reset()
    adapter.step(0)
    t = adapter.telemetry()
    for k in _REQUIRED_KEYS:
        assert k in t, f"missing required telemetry key: {k!r}"


def test_telemetry_types():
    adapter = GymAdapter(_FakeGymEnv())
    adapter.reset()
    adapter.step(0)
    t = adapter.telemetry()
    assert isinstance(t["type"], str)
    assert isinstance(t["kills"], int)
    assert isinstance(t["coverage"], float)
    assert isinstance(t["health"], float)
    assert isinstance(t["length"], int)
