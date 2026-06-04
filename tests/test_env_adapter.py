"""Tests for doom.env_adapter — Phase 6 generalisation interface."""
from types import SimpleNamespace

import numpy as np
import pytest

from doom.env_adapter import DoomCampaignAdapter, GymAdapter, EnvAdapter, make_adapter, register_adapter


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


# ---------------------------------------------------------------------------
# DoomCampaignAdapter — regression: it must CONSTRUCT (factory thunk) and read the
# real nested info shape `{"map", "doom": {...}}`, not the flat keys it never produced.
# ---------------------------------------------------------------------------

class _FakeCampaignEnv:
    observation_space = _FakeObsSpace()
    action_space = _FakeActionSpace()
    button_names = ["FWD", "ATK"]

    def reset(self, **kwargs):
        return np.zeros(1), {"map": "MAP01", "doom": {}}

    def step(self, action):
        info = {"map": "MAP01", "doom": {
            "deltas": {"killcount": 1},
            "levels": {"health": 80.0, "ammo2": 25.0},
            "terminal": "exit",
            "coverage_cells": 12,
        }}
        return np.zeros(1), 1.0, True, False, info

    def close(self):
        pass


def test_doom_adapter_constructs_and_maps_real_info(monkeypatch):
    # make_campaign_env is a factory: it returns a thunk that builds the env.
    monkeypatch.setattr("doom.campaign.make_campaign_env",
                        lambda *a, **k: (lambda: _FakeCampaignEnv()))
    cfg = SimpleNamespace(maps=["MAP01"])
    ad = DoomCampaignAdapter(cfg)
    assert isinstance(ad, EnvAdapter)
    ad.reset()
    ad.step(0)
    t = ad.telemetry()
    assert t["type"] == "exit"          # read from info["doom"]["terminal"]
    assert t["map"] == "MAP01"
    assert t["kills"] == 1              # accumulated from per-step killcount delta
    assert t["coverage"] == 12
    assert t["health"] == 80.0
    assert t["length"] == 1
    assert ad.button_names() == ["FWD", "ATK"]


def test_doom_adapter_kills_accumulate_over_steps(monkeypatch):
    monkeypatch.setattr("doom.campaign.make_campaign_env",
                        lambda *a, **k: (lambda: _FakeCampaignEnv()))
    ad = DoomCampaignAdapter(SimpleNamespace(maps=["MAP01"]))
    ad.reset()
    ad.step(0); ad.step(0); ad.step(0)
    assert ad.telemetry()["kills"] == 3
    assert ad.telemetry()["length"] == 3
    ad.reset()                          # reset clears the accumulators
    assert ad.telemetry()["kills"] == 0
