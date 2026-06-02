"""Gymnasium wrapper around ViZDoom.

Each step emits, besides obs/reward, an `info["doom"]` dict with the counter deltas
and the instantaneous levels (health/ammo). That signal feeds the StatsTracker and,
ultimately, the Obsidian notes.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from doom.geometry import read_wall_segments
from instrumentation.game_vars import LEVELS, MONOTONIC, TRACKED_VARS, VAR_NAMES

# --- Reward shaping (hits/misses and performance loss) ---
# Reward for a shot that LANDED (HITCOUNT delta within the frame_skip window).
HIT_REWARD = 1.0
# Penalty for ATTACKING and hitting nothing. Smaller than HIT_REWARD on purpose:
# too high a penalty teaches the agent to stop shooting (becomes passive).
MISS_PENALTY = 0.25
# Penalty for LOSING PERFORMANCE: taking damage (per point) and dying.
DAMAGE_TAKEN_PENALTY = 0.05
DEATH_PENALTY = 5.0


class DoomEnv(gym.Env):
    """Single-process env. Use the `make_doom_env` factory with SubprocVecEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "defend_the_center",
        frame_skip: int = 4,
        resolution: Tuple[int, int] = (84, 84),
        window_visible: bool = False,
        rewards: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.width, self.height = resolution
        r = rewards or {}
        self._hit_reward = float(r.get("hit_reward", HIT_REWARD))
        self._miss_penalty = float(r.get("miss_penalty", MISS_PENALTY))
        self._damage_penalty = float(r.get("damage_taken_penalty", DAMAGE_TAKEN_PENALTY))
        self._death_penalty = float(r.get("death_penalty", DEATH_PENALTY))
        self._move_reward = float(r.get("move_reward", 0.0))  # anti-idle (per distance)

        game = vzd.DoomGame()
        cfg = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        game.load_config(cfg)
        game.set_window_visible(window_visible)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        # A visible window needs a larger resolution to actually be watchable.
        game.set_screen_resolution(
            vzd.ScreenResolution.RES_640X480
            if window_visible
            else vzd.ScreenResolution.RES_160X120
        )
        # Override the cfg variables with our rich set.
        game.set_available_game_variables(TRACKED_VARS)
        game.set_sectors_info_enabled(True)  # map geometry for the real minimap
        game.init()
        self.game = game
        self._walls_pending = True  # send the walls once (fixed map in a scenario)

        self.buttons: List[vzd.Button] = game.get_available_buttons()
        self.button_names: List[str] = [b.name for b in self.buttons]
        n = len(self.buttons)
        # Discrete one-hot actions: one action per available button.
        self.actions: List[List[int]] = [
            [1 if i == j else 0 for i in range(n)] for j in range(n)
        ]

        # Index of the ATTACK button (to know when the agent "missed" a shot).
        self._attack_idx = next(
            (i for i, nm in enumerate(self.button_names) if "ATTACK" in nm.upper()),
            None,
        )

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )
        self._last_vars: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    def _read_raw_vars(self) -> Dict[str, float]:
        state = self.game.get_state()
        if state is None:
            return self._last_vars or {n: 0.0 for n in VAR_NAMES}
        vals = state.game_variables
        return {VAR_NAMES[i]: float(vals[i]) for i in range(len(VAR_NAMES))}

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        frame = state.screen_buffer  # (120, 160) uint8
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]

    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self._last_vars = self._read_raw_vars()
        self._ep_base = 0.0  # native (unshaped) scenario return, for fair A/B eval
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        buttons = self.actions[int(action)]
        base_reward = self.game.make_action(buttons, self.frame_skip)
        self._ep_base += base_reward
        done = self.game.is_episode_finished()

        if not done:
            raw = self._read_raw_vars()
            deltas = {n: max(0.0, raw[n] - self._last_vars[n]) for n in MONOTONIC}
            # Distance traveled this step (to measure map exploration).
            dx = raw["position_x"] - self._last_vars["position_x"]
            dy = raw["position_y"] - self._last_vars["position_y"]
            deltas["distance"] = float((dx * dx + dy * dy) ** 0.5)
            levels = {n: raw[n] for n in LEVELS}
            self._last_vars = raw
            obs = self._get_obs()
        else:
            # The terminal state has no screen/vars; use the last known ones.
            deltas = {n: 0.0 for n in MONOTONIC}
            deltas["distance"] = 0.0
            levels = {n: self._last_vars[n] for n in LEVELS}
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Shaping: + for a hit; - for a miss; - for losing performance (damage/death).
        reward = base_reward + self._hit_reward * deltas["hitcount"]
        attacked = self._attack_idx is not None and int(action) == self._attack_idx
        if attacked and deltas["hitcount"] == 0 and not done:
            reward -= self._miss_penalty
        reward -= self._damage_penalty * deltas["damage_taken"]
        reward += self._move_reward * deltas["distance"]  # reward moving (anti-idle)
        if done and self.game.is_player_dead():
            reward -= self._death_penalty

        doom = {"deltas": deltas, "levels": levels, "action": int(action)}
        if done:
            # Reliable terminal type: health reported is the pre-death frame, so use
            # ViZDoom's is_player_dead() instead of checking health<=0.
            doom["terminal"] = "death" if self.game.is_player_dead() else "timeout"
            doom["base_return"] = self._ep_base  # native episode return (no shaping)
        # Map geometry: sent ONCE (not every step — doesn't weigh on the loop).
        if self._walls_pending:
            doom["walls"] = read_wall_segments(self.game)
            self._walls_pending = False
        return obs, float(reward), done, False, {"doom": doom}

    def close(self) -> None:
        self.game.close()


def make_doom_env(
    scenario: str,
    frame_skip: int,
    resolution: Tuple[int, int],
    seed: int,
    rank: int,
    window_visible: bool = False,
    rewards: Optional[Dict[str, float]] = None,
):
    """Factory for SubprocVecEnv. Each subprocess gets a distinct seed.

    We don't wrap with Monitor here: VecMonitor (applied once over the vec env)
    already injects info["episode"], avoiding the duplicate-Monitor warning.
    """

    def _init():
        env = DoomEnv(
            scenario=scenario,
            frame_skip=frame_skip,
            resolution=resolution,
            window_visible=window_visible,
            rewards=rewards,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def probe_env_metadata(
    scenario: str, frame_skip: int, resolution: Tuple[int, int]
) -> Dict[str, Any]:
    """Create a temporary env just to discover button names and action count.

    Useful for the StatsTracker (action-distribution labels) without booting the
    whole training.
    """
    env = DoomEnv(scenario=scenario, frame_skip=frame_skip, resolution=resolution)
    meta = {
        "button_names": list(env.button_names),
        "num_actions": int(env.action_space.n),
    }
    env.close()
    return meta
