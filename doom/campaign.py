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

# Curated COMBINED actions (each presses MULTIPLE buttons at once). A naive one-hot space
# (one button per step) can't express Doom's core move-AND-shoot, so the agent collapsed
# into a defensive "back away + spray blindly" local optimum and got 0 kills (a RANDOM
# agent kills more). These combos always step FORWARD for locomotion and include NO
# MOVE_BACKWARD at all — removing the retreat optimum and forcing engagement — while still
# allowing turn-in-place to aim and a plain shoot. (combo button-names, short label)
CAMPAIGN_ACTIONS = [
    (["MOVE_FORWARD"], "FWD"),
    (["MOVE_FORWARD", "TURN_LEFT"], "FWD+TL"),
    (["MOVE_FORWARD", "TURN_RIGHT"], "FWD+TR"),
    (["MOVE_FORWARD", "ATTACK"], "FWD+ATK"),
    (["MOVE_FORWARD", "TURN_LEFT", "ATTACK"], "FWD+TL+ATK"),
    (["MOVE_FORWARD", "TURN_RIGHT", "ATTACK"], "FWD+TR+ATK"),
    (["TURN_LEFT"], "TL"),
    (["TURN_RIGHT"], "TR"),
    (["ATTACK"], "ATK"),
    (["MOVE_FORWARD", "USE"], "FWD+USE"),
    (["SELECT_NEXT_WEAPON"], "NEXTW"),
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
        spatial_memory: bool = False,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.width, self.height = resolution
        self.kills_to_clear = kills_to_clear
        self._spatial = bool(spatial_memory)  # 2nd obs channel: where I've been
        r = rewards or {}
        self._hit_reward = float(r.get("hit_reward", HIT_REWARD))
        self._miss_penalty = float(r.get("miss_penalty", MISS_PENALTY))
        self._damage_penalty = float(r.get("damage_taken_penalty", 0.1))
        self._move_reward = float(r.get("move_reward", 0.0))  # anti-idle (per distance)
        # Anti-circle: reward NET outward progress (new max distance from spawn), which
        # spinning in place CANNOT farm — unlike raw move_reward, which literally pays the
        # agent to drive in circles. Drives directed exploration instead of a limit cycle.
        self._frontier_reward = float(r.get("frontier_reward", 0.0))
        self._spawn_xy: Optional[Tuple[float, float]] = None
        self._max_dist = 0.0
        # Kill bonus was hardcoded at 5.0 — i.e. the single strongest combat lever wasn't
        # tunable, so the supervisor/suggestions could never push the agent OUT of the
        # combat-avoidance local optimum (idle-to-timeout beats fight-and-die). Now a knob.
        self._kill_reward = float(r.get("kill_reward", 5.0))
        # Per-monster kill multipliers learned from the bestiary (deadlier = worth more).
        self._threat_mult: Dict[str, float] = dict(r.get("enemy_threat", {}) or {})
        self._step_threat_bonus = 0.0  # extra kill reward this step from threat weighting
        self._living_reward = float(r.get("living_reward", 0.01))
        # Death penalty was hardcoded to 100 -> with kills worth only 5, the optimal
        # deterministic policy was COWARDICE (idle to timeout beats fighting+dying).
        # Now configurable and low so a kill is worth the risk of dying.
        self._death_penalty = float(r.get("death_penalty", 5.0))
        # Exploration & completion (autonomy goal: cover the map, reach the exit).
        self._coverage_reward = float(r.get("coverage_reward", 0.0))  # per NEW cell
        self._coverage_cell = float(r.get("coverage_cell", 96.0))     # grid size
        self._exit_reward = float(r.get("exit_reward", 0.0))          # reach the exit
        # Count-based weapon variety: bonus the first time a NEW weapon slot is wielded
        # this episode -> a reason to actually use SELECT_NEXT_WEAPON on what it picks up.
        self._weapon_variety_reward = float(r.get("weapon_variety_reward", 0.0))
        self._episode_timeout = int(episode_timeout)
        self._visited: set = set()  # grid cells seen this episode
        self._weapons_seen: set = set()  # weapon slots wielded this episode
        self._ep_enemies: Dict[str, Dict[str, Any]] = {}  # per-monster encounter facts
        self._mon_counts: Dict[str, int] = {}             # prev-step counts (kill detect)
        self._nearest_enemy: Optional[str] = None         # for "who killed the agent"
        self._ep_ticks = 0          # ticks elapsed this episode (exit vs timeout)
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
        game.set_objects_info_enabled(True)  # actor names/positions -> factual bestiary
        # Living reward < 0 = time penalty (anti-camping); dying is bad.
        game.set_living_reward(self._living_reward)
        game.set_death_penalty(self._death_penalty)
        game.init()
        self.game = game

        # Build one button-vector per COMBINED action; `button_names` carries one label
        # PER ACTION (the StatsTracker indexes its action_distribution by the discrete
        # action, so its labels must align 1:1 with the action set, not the raw buttons).
        bidx = {b.name: i for i, b in enumerate(CAMPAIGN_BUTTONS)}
        n_buttons = len(CAMPAIGN_BUTTONS)
        self.actions: List[List[int]] = []
        self.button_names: List[str] = []
        self._action_attacks: List[bool] = []  # does this action press ATTACK?
        for combo, label in CAMPAIGN_ACTIONS:
            vec = [0] * n_buttons
            for name in combo:
                vec[bidx[name]] = 1
            self.actions.append(vec)
            self.button_names.append(label)
            self._action_attacks.append(bool(vec[bidx["ATTACK"]]))

        self.action_space = spaces.Discrete(len(self.actions))
        channels = 2 if self._spatial else 1
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, channels), dtype=np.uint8
        )
        self._last_vars: Optional[Dict[str, float]] = None
        self._walls_pending = True  # send walls once per map (re-armed on switch)
        # Spatial memory: a persistent visited-grid rendered as the 2nd channel.
        self._visit_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self._bbox: Optional[Tuple[float, float, float, float]] = None  # map extent

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
            frame = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            frame = cv2.resize(
                state.screen_buffer, (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        if not self._spatial:
            return frame[:, :, None]
        # 2nd channel: the agent's own memory of where it has already been.
        return np.stack([frame, self._visit_grid], axis=2)

    # ------------------------------------------------------------------
    def _compute_bbox(self) -> None:
        """Map extent (from wall geometry) to project world coords into the grid."""
        try:
            walls = read_wall_segments(self.game)
        except Exception:
            walls = []
        if walls:
            xs = [c for w in walls for c in (w[0], w[2])]
            ys = [c for w in walls for c in (w[1], w[3])]
            self._bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            self._bbox = None

    def _world_to_px(self, x: float, y: float):
        if not self._bbox:
            return None
        minx, miny, maxx, maxy = self._bbox
        sx = (maxx - minx) or 1.0
        sy = (maxy - miny) or 1.0
        px = int((x - minx) / sx * (self.width - 1))
        py = int((maxy - y) / sy * (self.height - 1))  # flip Y for image space
        if 0 <= px < self.width and 0 <= py < self.height:
            return px, py
        return None

    _ENGAGE_RANGE = 320.0  # map units: within this counts as "engaged" (for approach stats)

    def _track_enemies(self, px: float, py: float, weapon: int, kills_step: int) -> None:
        """FACTUAL per-monster tracking from the (map-wide) objects buffer. Because object
        counts are exact, a drop in a type's count on a step the agent scored a kill =
        the agent killed THAT type (with the current weapon). Also tracks the nearest
        monster (for death attribution) and 'ranged' (projectile/hitscan). Cheap."""
        state = self.game.get_state()
        if state is None or not state.objects:
            return
        from doom.entities import MONSTERS, PROJECTILE_CASTER
        cur: Dict[str, int] = {}
        nearest, nearest_d2 = None, 1e18
        for o in state.objects:
            name = o.name
            caster = PROJECTILE_CASTER.get(name)
            if caster:  # a projectile in flight -> its caster is a ranged attacker
                self._ep_enemies.setdefault(caster, {})["ranged"] = True
                continue
            if name not in MONSTERS:
                continue
            cur[name] = cur.get(name, 0) + 1
            dx, dy = px - o.position_x, py - o.position_y
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2, nearest = d2, name
            e = self._ep_enemies.setdefault(name, {})
            e["total"] = max(e.get("total", 0), cur[name])  # spawn count on this map
            if d2 <= self._ENGAGE_RANGE * self._ENGAGE_RANGE:  # only engaged monsters
                e["seen"] = e.get("seen", 0) + 1
                e["approach"] = e.get("approach", 0) + (
                    1 if (o.velocity_x * dx + o.velocity_y * dy) > 0.0 else 0)
                e["dist_min"] = min(e.get("dist_min", 1e9), d2 ** 0.5)
        # Exact kill-by-type: a type whose count dropped on a kill step was killed by us.
        self._step_threat_bonus = 0.0
        if kills_step > 0 and self._mon_counts:
            for name, prev in self._mon_counts.items():
                drop = prev - cur.get(name, 0)
                if drop > 0:
                    e = self._ep_enemies.setdefault(name, {})
                    e["killed"] = e.get("killed", 0) + drop
                    kw = e.setdefault("kill_weapon", {})
                    kw[weapon] = kw.get(weapon, 0) + drop
                    # Bestiary -> reward: pay extra for deadlier types (mult-1 on top of flat).
                    mult = self._threat_mult.get(name, 1.0)
                    if mult > 1.0:
                        self._step_threat_bonus += self._kill_reward * (mult - 1.0) * drop
        self._mon_counts = cur
        self._nearest_enemy = nearest  # remember for "who killed the agent"

    def _mark_visit(self, x: float, y: float) -> None:
        if not self._spatial:
            return
        p = self._world_to_px(x, y)
        if p:
            px, py = p
            # 3x3 footprint so the trail is visible to the CNN.
            self._visit_grid[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = 255

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
        self._ep_ticks = 0
        self._visited = set()
        # Seed with the spawn weapon so only switching to a DIFFERENT slot earns variety.
        self._weapons_seen = {int(self._last_vars.get("selected_weapon", 0.0))}
        self._ep_enemies = {}
        self._mon_counts = {}
        self._nearest_enemy = None
        self._spawn_xy = (self._last_vars["position_x"], self._last_vars["position_y"])
        self._max_dist = 0.0
        self._visit_grid[:] = 0
        if self._spatial:
            self._compute_bbox()
            self._mark_visit(self._last_vars["position_x"], self._last_vars["position_y"])
        return self._get_obs(), {"map": self._current_map}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        buttons = self.actions[int(action)]
        base_reward = self.game.make_action(buttons, self.frame_skip)
        self._ep_base += base_reward
        self._ep_ticks += self.frame_skip
        done = self.game.is_episode_finished()

        cov_bonus = 0.0  # reward for stepping on a NEW grid cell this episode
        weapon_bonus = 0.0  # reward the first time a NEW weapon slot is wielded
        frontier_bonus = 0.0  # reward NET outward progress (can't be farmed by circling)
        if not done:
            raw = self._read_raw_vars()
            deltas = {n: max(0.0, raw[n] - self._last_vars[n]) for n in MONOTONIC}
            dx = raw["position_x"] - self._last_vars["position_x"]
            dy = raw["position_y"] - self._last_vars["position_y"]
            deltas["distance"] = float((dx * dx + dy * dy) ** 0.5)
            levels = {n: raw[n] for n in LEVELS}
            self._last_vars = raw
            # Count-based exploration: bonus only the FIRST time a cell is visited.
            if self._coverage_reward and self._coverage_cell > 0:
                cell = (round(raw["position_x"] / self._coverage_cell),
                        round(raw["position_y"] / self._coverage_cell))
                if cell not in self._visited:
                    self._visited.add(cell)
                    cov_bonus = self._coverage_reward
            # Weapon variety: reward wielding a slot not yet used this episode (only
            # real weapons, slot >= 2 — slot 1 is the fist/pistol it always starts with).
            if self._weapon_variety_reward:
                slot = int(raw.get("selected_weapon", 0.0))
                if slot >= 2 and slot not in self._weapons_seen:
                    self._weapons_seen.add(slot)
                    weapon_bonus = self._weapon_variety_reward
            # Frontier progress: reward only when the agent reaches a NEW farthest point
            # from spawn. Circling never increases the max distance -> earns nothing.
            if self._frontier_reward and self._spawn_xy is not None:
                ddx = raw["position_x"] - self._spawn_xy[0]
                ddy = raw["position_y"] - self._spawn_xy[1]
                dist = (ddx * ddx + ddy * ddy) ** 0.5
                if dist > self._max_dist:
                    frontier_bonus = self._frontier_reward * (dist - self._max_dist)
                    self._max_dist = dist
            self._mark_visit(raw["position_x"], raw["position_y"])
            self._track_enemies(raw["position_x"], raw["position_y"],
                                int(raw.get("selected_weapon", 0)),
                                int(deltas.get("killcount", 0)))
            obs = self._get_obs()
        else:
            deltas = {n: 0.0 for n in MONOTONIC}
            deltas["distance"] = 0.0
            levels = {n: self._last_vars[n] for n in LEVELS}
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            self._step_threat_bonus = 0.0  # no tracking on the terminal step

        # Reward shaping: kills positive, damage taken negative, + aim, + movement,
        # + exploration (new cells).
        shaped = base_reward + self._kill_reward * deltas["killcount"]
        shaped -= self._damage_penalty * deltas["damage_taken"]
        shaped += self._hit_reward * deltas["hitcount"]
        shaped += self._move_reward * deltas["distance"]  # reward moving (anti-idle)
        shaped += cov_bonus                               # reward exploring new ground
        shaped += weapon_bonus                            # reward using a new weapon
        shaped += frontier_bonus                          # reward NET outward progress (anti-circle)
        shaped += self._step_threat_bonus                 # extra for deadlier monsters (bestiary)
        attacked = self._action_attacks[int(action)]  # this action pressed ATTACK
        if attacked and deltas["hitcount"] == 0 and not done:
            shaped -= self._miss_penalty

        # Completion: reaching the level EXIT = episode ended, not dead, before the
        # timeout. That's the real "I finished the map". Clearing the kill quota also
        # counts as a success (some maps gate the exit behind killing).
        dead = self.game.is_player_dead()
        reached_exit = bool(done and not dead and self._ep_ticks < self._episode_timeout)
        kills_total = self._last_vars.get("killcount", 0.0) if self._last_vars else 0.0
        cleared = kills_total >= self.kills_to_clear
        success = reached_exit or cleared
        if done:
            if reached_exit:
                shaped += self._exit_reward   # the big prize: finishing the level
            elif cleared:
                shaped += 100.0               # cleared the enemies

        doom = {
            "deltas": deltas,
            "levels": levels,
            "action": int(action),
            "attacked": bool(attacked),  # tracker counts attacks from this (combined actions)
            "success": success,
        }
        if done:
            doom["terminal"] = (
                "death" if dead else ("exit" if reached_exit else "timeout")
            )
            doom["base_return"] = self._ep_base  # native episode return (no shaping)
            doom["coverage_cells"] = len(self._visited)  # for frontier curriculum
            # The actual cells visited this episode -> persistent per-map heatmap memory.
            # Once per episode (off the hot path), so within the ±2% budget.
            doom["visited_cells"] = [[gx, gy] for (gx, gy) in self._visited]
            # Who killed the agent: the nearest monster at the moment of death (proxy).
            if dead and self._nearest_enemy:
                self._ep_enemies.setdefault(self._nearest_enemy, {})["killed_agent"] = 1
            doom["enemies"] = self._ep_enemies  # per-monster facts -> the bestiary
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
    spatial_memory: bool = False,
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
            spatial_memory=spatial_memory,
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
