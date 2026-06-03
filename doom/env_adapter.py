"""Env Adapter Interface (Phase 6) — thin contract so any environment plugs into
HeLLMind's cognitive loop without changing the cognition code.

The adapter provides:
  - obs_space / action_space (for SB3)
  - step / reset (standard Gym API)
  - telemetry() → dict  (what gets recorded into the cognitive memory)
  - button_names → List[str] (for StatsTracker)

Concrete adapters:
  - DoomCampaignAdapter  (wraps doom/campaign.py — already the production path)
  - GymAdapter           (wraps any gym.Env — the generalisation bridge)

The cognition layer (memory, behavior, hypothesize, experiment, curriculum)
only ever calls `adapter.telemetry()` and standard Gym methods, so swapping
Doom for MuJoCo requires only a new adapter, not new cognition code.

    from doom.env_adapter import GymAdapter
    import gymnasium as gym
    adapter = GymAdapter(gym.make("CartPole-v1"))
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EnvAdapter(ABC):
    """Thin contract: standard Gym step/reset + HeLLMind telemetry."""

    # --- Gym-compatible ---

    @property
    @abstractmethod
    def observation_space(self): ...

    @property
    @abstractmethod
    def action_space(self): ...

    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]: ...

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]: ...

    @abstractmethod
    def close(self) -> None: ...

    # --- HeLLMind-specific ---

    @abstractmethod
    def button_names(self) -> List[str]:
        """Action labels for StatsTracker."""
        ...

    @abstractmethod
    def telemetry(self) -> Dict[str, Any]:
        """End-of-episode telemetry dict written to cognitive memory.

        Required keys (cognition code reads these):
          type      : str   — 'death' | 'success' | 'exit' | 'timeout'
          map       : str   — level / environment name
          kills     : int   — enemies eliminated (0 for non-combat envs)
          coverage  : float — fraction of map/state space visited (0.0–1.0)
          health    : float — agent health fraction at episode end (0.0–1.0)
          length    : int   — episode length in steps
        Optional keys (richer cognition):
          ammo, accuracy, enemies_seen, weapon, region
        """
        ...


# ---------------------------------------------------------------------------
# Doom Campaign adapter (wraps the existing doom/campaign.py env)
# ---------------------------------------------------------------------------

class DoomCampaignAdapter(EnvAdapter):
    """Thin wrapper around the ViZDoom campaign environment.

    This is the *existing* production path — DoomCampaignEnv already provides
    all the telemetry; this adapter just exposes it under the standard contract.
    """

    def __init__(self, cfg) -> None:
        from doom.campaign import make_campaign_env
        self._env = make_campaign_env(cfg)
        self._last_info: dict = {}

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self._last_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def button_names(self) -> List[str]:
        return getattr(self._env, "button_names", [])

    def telemetry(self) -> Dict[str, Any]:
        info = self._last_info
        return {
            "type":     info.get("terminal_reason", "timeout"),
            "map":      info.get("map", ""),
            "kills":    int(info.get("kills", 0)),
            "coverage": float(info.get("coverage_fraction", 0.0)),
            "health":   float(info.get("health_fraction", 0.0)),
            "length":   int(info.get("length", 0)),
            "ammo":     float(info.get("ammo", 0.0)),
        }


# ---------------------------------------------------------------------------
# Generic Gymnasium adapter
# ---------------------------------------------------------------------------

class GymAdapter(EnvAdapter):
    """Wraps any gymnasium.Env so it works with HeLLMind's cognition pipeline.

    For non-combat environments, kills=0 and coverage is approximated from the
    fraction of distinct observations seen (requires discrete obs or a hash).
    Subclass this and override `telemetry()` for richer environment-specific data.
    """

    def __init__(self, env, env_name: str = "") -> None:
        self._env = env
        self._name = env_name or getattr(env, "spec", None) and env.spec.id or "env"
        self._episode_length = 0
        self._terminated = False
        self._truncated = False
        self._total_reward = 0.0
        self._obs_hashes: set = set()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, **kwargs):
        self._episode_length = 0
        self._terminated = False
        self._truncated = False
        self._total_reward = 0.0
        self._obs_hashes.clear()
        obs, info = self._env.reset(**kwargs)
        self._track_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._episode_length += 1
        self._total_reward += float(reward)
        self._terminated = terminated
        self._truncated = truncated
        self._track_obs(obs)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def button_names(self) -> List[str]:
        n = self._env.action_space.n if hasattr(self._env.action_space, "n") else 1
        return [f"action_{i}" for i in range(n)]

    def telemetry(self) -> Dict[str, Any]:
        if self._terminated:
            outcome = "success"
        elif self._truncated:
            outcome = "timeout"
        else:
            outcome = "timeout"
        return {
            "type":     outcome,
            "map":      self._name,
            "kills":    0,
            "coverage": self._coverage_fraction(),
            "health":   1.0 if not self._terminated else 0.0,
            "length":   self._episode_length,
        }

    # Private

    def _track_obs(self, obs: np.ndarray) -> None:
        try:
            self._obs_hashes.add(hash(obs.tobytes()))
        except (AttributeError, TypeError):
            pass

    def _coverage_fraction(self) -> float:
        obs_space = self._env.observation_space
        if hasattr(obs_space, "n"):
            return min(1.0, len(self._obs_hashes) / obs_space.n)
        return min(1.0, len(self._obs_hashes) / max(1, self._episode_length * 2))


# ---------------------------------------------------------------------------
# Registry (optional convenience)
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    "doom_campaign": DoomCampaignAdapter,
    "gym":           GymAdapter,
}


def make_adapter(adapter_type: str, *args, **kwargs) -> EnvAdapter:
    cls = _REGISTRY.get(adapter_type)
    if cls is None:
        raise ValueError(f"Unknown adapter: {adapter_type!r}. Available: {list(_REGISTRY)}")
    return cls(*args, **kwargs)


def register_adapter(name: str, cls: type) -> None:
    """Register a custom adapter so it can be created via make_adapter()."""
    _REGISTRY[name] = cls
