"""CAMPAIGN mode: play full WAD maps (Doom 1 / Freedoom), in order.

Unlike the ViZDoom scenarios (single, short objective), here we load a WAD with real
maps (MAP01.., or E1M1.. in the original doom.wad) and train the agent to "complete
and move on to the next".

Design decisions:
- "Complete" = survive AND/OR kill X enemies (success criterion, used in reward
  shaping and logging).
- Advancing between maps = SEQUENTIAL BY TIMESTEPS (see MapCurriculumCallback).

The env emits `info["doom"]` in the SAME format as the scenario env (deltas/levels/
action) to reuse the StatsTracker, plus `info["map"]` and `info["doom"]["success"]`.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from doom.env import HIT_REWARD, MISS_PENALTY
from doom.geometry import read_wall_segments
from instrumentation.game_vars import LEVELS, MONOTONIC, TRACKED_VARS, VAR_NAMES

# Buttons needed to TRAVERSE a map (move, turn, shoot, open doors).
CAMPAIGN_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.USE,                  # open doors / trigger the level exit
    vzd.Button.SPEED,
    vzd.Button.SELECT_NEXT_WEAPON,   # switch weapons (creates weapon variety)
]


def default_wad() -> str:
    """Default WAD: the freedoom2.wad bundled with ViZDoom (free/legal).

    It lives at the root of the vizdoom package (next to scenarios/), not inside
    scenarios/.
    """
    return os.path.join(os.path.dirname(vzd.scenarios_path), "freedoom2.wad")


class CampaignDoomEnv(gym.Env):
    """Plays a full WAD map. The map can be switched at runtime."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        wad_path: str,
        doom_map: str = "MAP01",
        frame_skip: int = 4,
        resolution: Tuple[int, int] = (84, 84),
        episode_timeout: int = 2100,   # ticks (~60s at 35fps) so it doesn't hang
        kills_to_clear: int = 5,
        window_visible: bool = False,
        rewards: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.width, self.height = resolution
        self.kills_to_clear = kills_to_clear
        r = rewards or {}
        self._hit_reward = float(r.get("hit_reward", HIT_REWARD))
        self._miss_penalty = float(r.get("miss_penalty", MISS_PENALTY))
        self._damage_penalty = float(r.get("damage_taken_penalty", 0.1))
        self._current_map = doom_map
        self._pending_map: Optional[str] = None

        game = vzd.DoomGame()
        # Full IWAD (freedoom2.wad / doom.wad) -> game path; maps via set_doom_map.
        game.set_doom_game_path(wad_path)
        game.set_doom_map(doom_map)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(
            vzd.ScreenResolution.RES_640X480
            if window_visible
            else vzd.ScreenResolution.RES_160X120
        )
        game.set_window_visible(window_visible)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_episode_timeout(episode_timeout)
        game.set_available_buttons(CAMPAIGN_BUTTONS)
        game.set_available_game_variables(TRACKED_VARS)
        game.set_sectors_info_enabled(True)  # map geometry for the real minimap
        # ViZDoom's built-in rewards: being alive is good, dying is bad.
        game.set_living_reward(0.01)
        game.set_death_penalty(100.0)
        game.init()
        self.game = game

        self.button_names: List[str] = [b.name for b in CAMPAIGN_BUTTONS]
        n = len(CAMPAIGN_BUTTONS)
        self.actions: List[List[int]] = [
            [1 if i == j else 0 for i in range(n)] for j in range(n)
        ]
        self._attack_idx = next(
            (i for i, nm in enumerate(self.button_names) if "ATTACK" in nm.upper()),
            None,
        )

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )
        self._last_vars: Optional[Dict[str, float]] = None
        self._walls_pending = True  # send walls once per map (re-armed on switch)

    # ------------------------------------------------------------------
    def set_map(self, doom_map: str) -> None:
        """Schedule a map switch; applied on the next reset (used by the curriculum)."""
        self._pending_map = doom_map

    @property
    def current_map(self) -> str:
        return self._current_map

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
        frame = cv2.resize(
            state.screen_buffer, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]

    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        if self._pending_map is not None and self._pending_map != self._current_map:
            self.game.set_doom_map(self._pending_map)
            self._current_map = self._pending_map
            self._pending_map = None
            self._walls_pending = True  # new map -> resend the geometry
        self.game.new_episode()
        self._last_vars = self._read_raw_vars()
        self._ep_base = 0.0  # native (unshaped) return, for fair A/B eval
        return self._get_obs(), {"map": self._current_map}

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
            dx = raw["position_x"] - self._last_vars["position_x"]
            dy = raw["position_y"] - self._last_vars["position_y"]
            deltas["distance"] = float((dx * dx + dy * dy) ** 0.5)
            levels = {n: raw[n] for n in LEVELS}
            self._last_vars = raw
            obs = self._get_obs()
        else:
            deltas = {n: 0.0 for n in MONOTONIC}
            deltas["distance"] = 0.0
            levels = {n: self._last_vars[n] for n in LEVELS}
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Reward shaping: kills positive, damage taken negative, + aim.
        shaped = base_reward + 5.0 * deltas["killcount"]
        shaped -= self._damage_penalty * deltas["damage_taken"]
        shaped += self._hit_reward * deltas["hitcount"]
        attacked = self._attack_idx is not None and int(action) == self._attack_idx
        if attacked and deltas["hitcount"] == 0 and not done:
            shaped -= self._miss_penalty

        # "Completed" = the episode ended without dying (reached the exit / survived
        # the timeout) OR already hit the kill quota in the episode.
        alive = levels["health"] > 0
        kills_total = self._last_vars.get("killcount", 0.0) if self._last_vars else 0.0
        success = bool(done and alive) or (kills_total >= self.kills_to_clear)
        if done and alive:
            shaped += 100.0  # bonus for completing the map alive

        doom = {
            "deltas": deltas,
            "levels": levels,
            "action": int(action),
            "success": success,
        }
        if done:
            doom["terminal"] = (
                "death" if self.game.is_player_dead()
                else ("success" if success else "timeout")
            )
            doom["base_return"] = self._ep_base  # native episode return (no shaping)
        if self._walls_pending:
            doom["walls"] = read_wall_segments(self.game)
            self._walls_pending = False
        return obs, float(shaped), done, False, {"map": self._current_map, "doom": doom}

    def close(self) -> None:
        self.game.close()


def make_campaign_env(
    wad_path: str,
    doom_map: str,
    frame_skip: int,
    resolution: Tuple[int, int],
    episode_timeout: int,
    kills_to_clear: int,
    seed: int,
    rank: int,
    window_visible: bool = False,
    rewards: Optional[Dict[str, float]] = None,
):
    """Factory for SubprocVecEnv/DummyVecEnv."""

    def _init():
        env = CampaignDoomEnv(
            wad_path=wad_path,
            doom_map=doom_map,
            frame_skip=frame_skip,
            resolution=resolution,
            episode_timeout=episode_timeout,
            kills_to_clear=kills_to_clear,
            window_visible=window_visible,
            rewards=rewards,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def campaign_metadata(wad_path: str, doom_map: str) -> Dict[str, Any]:
    """Discover button names / action count without booting training."""
    env = CampaignDoomEnv(wad_path=wad_path, doom_map=doom_map)
    meta = {"button_names": list(env.button_names), "num_actions": int(env.action_space.n)}
    env.close()
    return meta
