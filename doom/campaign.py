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
import random
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
    vzd.Button.MOVE_LEFT,            # strafe left  (opt-in actions; harmless if unused)
    vzd.Button.MOVE_RIGHT,           # strafe right
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

# Extra STRAFE actions (opt-in): sideways movement for dodging + navigation. Doom's strafe
# is a core movement the turn-only set can't express. Appended after the base set so the
# base action indices are unchanged (action count differs -> the brain name's `a{N}` keeps
# strafe and non-strafe brains from cross-loading).
STRAFE_ACTIONS = [
    (["MOVE_LEFT"], "SL"),
    (["MOVE_RIGHT"], "SR"),
    (["MOVE_FORWARD", "MOVE_LEFT"], "FWD+SL"),
    (["MOVE_FORWARD", "MOVE_RIGHT"], "FWD+SR"),
]


def campaign_actions(strafe: bool = False) -> list:
    """The action set, optionally with strafe combos appended."""
    return CAMPAIGN_ACTIONS + (STRAFE_ACTIONS if strafe else [])


def default_wad() -> str:
    """Default WAD: the freedoom2.wad bundled with ViZDoom (free/legal).

    It lives at the root of the vizdoom package (next to scenarios/), not inside
    scenarios/.
    """
    return os.path.join(os.path.dirname(vzd.scenarios_path), "freedoom2.wad")


def mode_scales(enemies_in_view: int, split_on: bool, factor: float):
    """Combat/exploration decoupling weights (pure, so it's unit-testable without ViZDoom).

    Returns (explore_scale, combat_scale):
      - split off  -> (1, 1): every shaping term at full strength.
      - enemy seen -> (factor, 1): COMBAT — damp exploration pulls, full combat signal.
      - clear view -> (1, factor): EXPLORE — full exploration, damp combat penalties.
    """
    if not split_on:
        return 1.0, 1.0
    if enemies_in_view > 0:
        return factor, 1.0
    return 1.0, factor


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
        memory_dir: Optional[str] = None,
        depth_perception: bool = False,
        strafe: bool = False,
        automap: bool = False,
        use_labels: bool = False,
        game_vars: bool = False,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.width, self.height = resolution
        self.kills_to_clear = kills_to_clear
        self._spatial = bool(spatial_memory)  # 2nd obs channel: where I've been
        self._depth = bool(depth_perception)  # extra obs channel: ViZDoom depth buffer
        self._strafe = bool(strafe)           # add sideways-movement actions
        self._automap = bool(automap)         # extra obs channel: native top-down automap
        self._use_labels = bool(use_labels)   # ground-truth on-screen enemy detection
        self._game_vars = bool(game_vars)     # feed HEALTH/AMMO into the policy (DFP/Arnold)
        self._engagement_reward = float((rewards or {}).get("engagement_reward", 0.0))
        r = rewards or {}
        # Combat/exploration decoupling: gate which objective the shaping emphasises by
        # ground-truth enemy visibility (needs labels). See Config.combat_explore_split.
        self._combat_explore_split = bool(r.get("combat_explore_split", 0.0))
        self._ce_factor = float(r.get("combat_explore_factor", 0.25))
        # Auto-USE: open doors / hit switches on contact (see Config.auto_use).
        self._auto_use = bool(r.get("auto_use", 0.0))
        self._use_idx: Optional[int] = None  # set once the button layout is built
        # Discovery reward: pay the first sighting of each new object per episode.
        self._discovery_reward = float(r.get("discovery_reward", 0.0))
        self._seen_objects: set = set()
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
        # Exit proximity reward: after the agent reaches the exit once, the position is
        # memorised and subsequent episodes get a small reward for getting closer to it.
        # Scale = fraction of exit_reward per unit of normalised progress (0→1).
        self._exit_prox_scale = float(r.get("exit_prox_scale", 0.1))
        # Exit memory persisted across runs/iterations/parallel envs. Without it, _exit_pos
        # lives only in THIS env instance for THIS subprocess — so an exit found by one env
        # never helps the others, and is forgotten the moment training ends (the eval and
        # the next auto-loop iteration are separate processes). The disk store fixes that:
        # the first env to reach the exit writes it; everyone reading the same vault gets it.
        self._exit_store = None
        if memory_dir:
            from writer.exit_store import ExitStore
            self._exit_store = ExitStore(memory_dir)
        self._exit_pos: Optional[Tuple[float, float]] = None   # memorised once per map
        self._prev_exit_dist: Optional[float] = None           # distance at last step
        self._spawn_exit_dist: Optional[float] = None          # spawn→exit dist (this episode)
        self._closest_exit_dist: Optional[float] = None        # closest the agent got to it
        # Go-Explore "return, then explore": frontier-goal archive + goal-conditioned reward.
        self._goal_prob = float(r.get("goexplore_goal_prob", 0.0))
        self._goal_scale = float(r.get("goexplore_goal_scale", 0.01))
        self._goal_radius = float(r.get("goexplore_reach_radius", 96.0))
        self._frontier_store = None
        if memory_dir and self._goal_prob > 0.0:
            from writer.frontier_store import FrontierStore
            self._frontier_store = FrontierStore(memory_dir, cell_size=self._coverage_cell)
        self._goal_xy: Optional[Tuple[float, float]] = None    # this episode's return target
        self._goal_reached = False
        self._prev_goal_dist: Optional[float] = None
        self._ep_positions: List[Tuple[float, float]] = []     # reached cells -> archive
        # Count-based weapon variety: bonus the first time a NEW weapon slot is wielded
        # this episode -> a reason to actually use SELECT_NEXT_WEAPON on what it picks up.
        self._weapon_variety_reward = float(r.get("weapon_variety_reward", 0.0))
        self._episode_timeout = int(episode_timeout)
        # Intrinsic curiosity (RND): spatial bonus that never saturates.
        use_rnd = float(r.get("use_rnd", 0.0)) > 0.0
        rnd_scale = float(r.get("rnd_scale", 0.5))
        if use_rnd:
            from doom.rnd import RNDModule
            self._rnd = RNDModule(rnd_scale=rnd_scale)
        else:
            self._rnd = None
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
        # Speed: skip rendering cosmetic elements the agent doesn't need to learn from. Each
        # disabled layer is render time saved every single frame (ViZDoom is CPU-render-bound).
        # Kept: the weapon sprite (useful context). Dropped: HUD/crosshair/decals/particles/
        # corpses/messages — pure clutter for a GRAY8 84×84 policy view.
        for setter, val in (("set_render_hud", False), ("set_render_minimal_hud", False),
                            ("set_render_crosshair", False), ("set_render_decals", False),
                            ("set_render_particles", False), ("set_render_corpses", False),
                            ("set_render_messages", False),
                            ("set_render_effects_sprites", False)):
            fn = getattr(game, setter, None)
            if fn is not None:
                try:
                    fn(val)
                except Exception:
                    pass  # tolerate ViZDoom builds missing a given setter
        game.set_mode(vzd.Mode.PLAYER)
        game.set_episode_timeout(episode_timeout)
        game.set_available_buttons(CAMPAIGN_BUTTONS)
        game.set_available_game_variables(TRACKED_VARS)
        game.set_sectors_info_enabled(True)  # map geometry for the real minimap
        game.set_objects_info_enabled(True)  # actor names/positions -> factual bestiary
        if self._depth:
            game.set_depth_buffer_enabled(True)  # per-pixel distance -> 3D structure channel
        if self._automap:
            game.set_automap_buffer_enabled(True)  # native top-down explored map channel
            game.set_automap_mode(vzd.AutomapMode.OBJECTS)  # walls + objects, no HUD clutter
            game.set_automap_rotate(False)         # keep it north-up (stable allocentric frame)
            game.set_automap_render_textures(False)  # clean line-art -> crisper for the CNN
        if self._use_labels:
            game.set_labels_buffer_enabled(True)   # ground-truth on-screen actor labels
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
        for combo, label in campaign_actions(self._strafe):
            vec = [0] * n_buttons
            for name in combo:
                vec[bidx[name]] = 1
            self.actions.append(vec)
            self.button_names.append(label)
            self._action_attacks.append(bool(vec[bidx["ATTACK"]]))
        self._use_idx = bidx.get("USE")  # for auto-USE (open doors on contact)

        self.action_space = spaces.Discrete(len(self.actions))
        channels = (1 + (1 if self._spatial else 0) + (1 if self._depth else 0)
                    + (1 if self._automap else 0))
        image_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, channels), dtype=np.uint8
        )
        # GAME_VARS_OBS: which numeric state to feed the policy, and how to normalise it to
        # ~[0,1]. HEALTH + AMMO are the decision-relevant ones (retreat when low, push when
        # full). Without these the agent is BLIND to its own health and keeps fighting until
        # it dies at low HP (the dominant death mode we measured).
        self._gamevar_specs = [("health", 100.0), ("ammo2", 50.0)]
        if self._game_vars:
            self.observation_space = spaces.Dict({
                "image": image_space,
                "vars": spaces.Box(low=0.0, high=1.0,
                                   shape=(len(self._gamevar_specs),), dtype=np.float32),
            })
        else:
            self.observation_space = image_space
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

    def _get_obs(self):
        img = self._get_image()
        if not self._game_vars:
            return img
        return {"image": img, "vars": self._gamevar_vector()}

    def _get_image(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None:
            frame = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            frame = cv2.resize(
                state.screen_buffer, (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        # Channels, in a FIXED order the obs space declares: pixels, [spatial], [depth], [automap].
        if not self._spatial and not self._depth and not self._automap:
            return frame[:, :, None]
        channels = [frame]
        if self._spatial:  # the agent's own memory of where it has already been
            channels.append(self._visit_grid)
        if self._depth:    # ViZDoom depth buffer (per-pixel distance) -> 3D structure
            channels.append(self._depth_frame(state))
        if self._automap:  # native top-down map of the explored layout
            channels.append(self._automap_frame(state))
        return np.stack(channels, axis=2)

    def _gamevar_vector(self) -> np.ndarray:
        """Normalised [health, ammo] in [0,1] for the policy — so it KNOWS its own state."""
        vars_ = self._last_vars or {}
        vec = [min(1.0, max(0.0, float(vars_.get(name, 0.0)) / scale))
               for name, scale in self._gamevar_specs]
        return np.asarray(vec, dtype=np.float32)

    def _zero_obs(self):
        """A valid all-zero observation (image, or Dict {image, vars}) for the terminal step."""
        img = np.zeros((self.height, self.width, self._image_channels()), dtype=np.uint8)
        if not self._game_vars:
            return img
        return {"image": img,
                "vars": np.zeros(len(self._gamevar_specs), dtype=np.float32)}

    def _image_channels(self) -> int:
        return (1 + (1 if self._spatial else 0) + (1 if self._depth else 0)
                + (1 if self._automap else 0))

    def _depth_frame(self, state) -> np.ndarray:
        """Resized depth buffer as a uint8 channel (0 = near, 255 = far)."""
        depth = getattr(state, "depth_buffer", None) if state is not None else None
        if depth is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        return cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def _automap_frame(self, state) -> np.ndarray:
        """Resized automap buffer as a grayscale uint8 channel (the explored top-down map)."""
        amap = getattr(state, "automap_buffer", None) if state is not None else None
        if amap is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        if amap.ndim == 3:  # automap is rendered in the screen format (may be 3-channel)
            amap = cv2.cvtColor(amap, cv2.COLOR_RGB2GRAY)
        return cv2.resize(amap, (self.width, self.height), interpolation=cv2.INTER_AREA)

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
            self._exit_pos = None       # new map = unknown exit position
        # Pull a persisted exit for this map (set by a prior run/env that reached it). Read
        # each reset so a parallel env that just discovered the exit benefits the others.
        if self._exit_pos is None and self._exit_store is not None:
            self._exit_pos = self._exit_store.load(self._current_map)
        self.game.new_episode()
        self._last_vars = self._read_raw_vars()
        self._ep_base = 0.0  # native (unshaped) return, for fair A/B eval
        self._ep_ticks = 0
        self._visited = set()
        # Seed with the spawn weapon so only switching to a DIFFERENT slot earns variety.
        self._weapons_seen = {int(self._last_vars.get("selected_weapon", 0.0))}
        self._seen_objects = set()  # discovery reward: new objects seen THIS episode
        self._ep_enemies = {}
        self._mon_counts = {}
        self._nearest_enemy = None
        self._spawn_xy = (self._last_vars["position_x"], self._last_vars["position_y"])
        self._max_dist = 0.0
        self._prev_exit_dist = None
        self._spawn_exit_dist = None
        self._closest_exit_dist = None
        if self._exit_pos is not None and self._spawn_xy is not None:
            sx, sy = self._spawn_xy
            ex, ey = self._exit_pos
            self._prev_exit_dist = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            # Baselines for the dense exit_progress metric (how close it gets to the exit).
            self._spawn_exit_dist = self._prev_exit_dist
            self._closest_exit_dist = self._prev_exit_dist
        # Go-Explore: archive the cells the LAST episode reached, then maybe pick a frontier
        # cell as this episode's "return" goal (dense reward guides the agent back to it).
        if self._frontier_store is not None:
            if self._ep_positions:
                try:
                    self._frontier_store.merge(self._current_map, self._ep_positions)
                except OSError:
                    pass
            self._ep_positions = []
        self._goal_xy = None
        self._goal_reached = False
        self._prev_goal_dist = None
        if self._frontier_store is not None and random.random() < self._goal_prob:
            goal = self._frontier_store.sample_goal(self._current_map, self._spawn_xy)
            if goal is not None:
                self._goal_xy = goal
                gx, gy = goal
                self._prev_goal_dist = ((gx - self._spawn_xy[0]) ** 2
                                        + (gy - self._spawn_xy[1]) ** 2) ** 0.5
        self._visit_grid[:] = 0
        if self._spatial:
            self._compute_bbox()
            self._mark_visit(self._last_vars["position_x"], self._last_vars["position_y"])
        return self._get_obs(), {"map": self._current_map}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        buttons = self.actions[int(action)]
        if self._auto_use and self._use_idx is not None and not buttons[self._use_idx]:
            # Hold USE every frame so doors open / switches fire on contact, without changing
            # the discrete action the policy chose (copy so the stored action vector is intact).
            buttons = list(buttons)
            buttons[self._use_idx] = 1
        base_reward = self.game.make_action(buttons, self.frame_skip)
        self._ep_base += base_reward
        self._ep_ticks += self.frame_skip
        done = self.game.is_episode_finished()

        cov_bonus = 0.0  # reward for stepping on a NEW grid cell this episode
        weapon_bonus = 0.0  # reward the first time a NEW weapon slot is wielded
        frontier_bonus = 0.0  # reward NET outward progress (can't be farmed by circling)
        engage_bonus = 0.0  # reward keeping a visible enemy centred (labels buffer)
        discovery_bonus = 0.0  # reward the first sighting of a new object this episode
        self._enemies_in_view = 0  # telemetry: how many enemies the agent can see now
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
            # Labels buffer: ground-truth on-screen enemy detection + engagement reward.
            if self._use_labels:
                from doom.entities import visible_enemies, visible_object_names
                st = self.game.get_state()
                labels = getattr(st, "labels", None) if st else None
                view = visible_enemies(labels, screen_width=float(self.width))
                self._enemies_in_view = view["count"]
                if self._engagement_reward and view["nearest_centered"] is not None:
                    # 1.0 when an enemy is dead-centred, 0 at the screen edge.
                    engage_bonus = self._engagement_reward * (1.0 - view["nearest_centered"])
                # Goal discovery: pay the FIRST sighting of each new object this episode
                # (keys/weapons/powerups/new monster types) — guides exploration to objectives.
                if self._discovery_reward:
                    new_objs = visible_object_names(labels) - self._seen_objects
                    if new_objs:
                        self._seen_objects |= new_objs
                        discovery_bonus = self._discovery_reward * len(new_objs)
            # Go-Explore: record reached position for the frontier archive (sampled cheaply).
            if self._frontier_store is not None and (self._ep_ticks % 8 == 0):
                self._ep_positions.append((raw["position_x"], raw["position_y"]))
            obs = self._get_obs()
        else:
            deltas = {n: 0.0 for n in MONOTONIC}
            deltas["distance"] = 0.0
            levels = {n: self._last_vars[n] for n in LEVELS}
            obs = self._zero_obs()  # terminal step: a valid zero obs (handles Dict too)
            self._step_threat_bonus = 0.0  # no tracking on the terminal step

        # Combat/exploration decoupling (champion-style, gated by ground-truth enemy
        # visibility): pursue ONE objective at a time so they don't fight each other.
        #   enemy on screen -> COMBAT: damp exploration pulls (don't wander off mid-fight)
        #   screen clear     -> EXPLORE: damp the miss penalty (blind shots while navigating
        #                       shouldn't be punished); exploration drives.
        split_on = self._combat_explore_split and self._use_labels and not done
        explore_scale, combat_scale = mode_scales(
            self._enemies_in_view, split_on, self._ce_factor)
        cov_bonus *= explore_scale
        weapon_bonus *= explore_scale
        frontier_bonus *= explore_scale
        discovery_bonus *= explore_scale

        # Reward shaping: kills positive, damage taken negative, + aim, + movement,
        # + exploration (new cells).
        shaped = base_reward + self._kill_reward * deltas["killcount"]
        shaped -= self._damage_penalty * deltas["damage_taken"]
        shaped += self._hit_reward * deltas["hitcount"]
        shaped += self._move_reward * deltas["distance"]  # reward moving (anti-idle)
        shaped += cov_bonus                               # reward exploring new ground
        shaped += weapon_bonus                            # reward using a new weapon
        shaped += frontier_bonus                          # reward NET outward progress (anti-circle)
        shaped += engage_bonus                            # reward keeping a visible enemy centred
        shaped += discovery_bonus                         # reward discovering new objects (keys/items)
        shaped += self._step_threat_bonus                 # extra for deadlier monsters (bestiary)
        if self._rnd and not done:
            rnd_bonus = self._rnd.bonus(raw["position_x"], raw["position_y"])
            shaped += rnd_bonus * explore_scale  # curiosity is an exploration pull
        # Exit proximity shaping: small reward for getting closer to the memorised exit.
        # Activates only after the first successful exit (self._exit_pos is set).
        if self._exit_pos is not None and not done and self._prev_exit_dist is not None:
            px, py = raw["position_x"], raw["position_y"]
            ex, ey = self._exit_pos
            cur_dist = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
            progress = self._prev_exit_dist - cur_dist  # positive = got closer
            if progress > 0:
                shaped += self._exit_prox_scale * progress * 0.001  # tiny per-unit bonus
            self._prev_exit_dist = cur_dist
            if self._closest_exit_dist is not None:
                self._closest_exit_dist = min(self._closest_exit_dist, cur_dist)
        # Go-Explore goal shaping: dense reward for returning to this episode's frontier
        # goal. Once within reach_radius the goal is "achieved" — reward stops and the agent
        # explores OUTWARD from that far launch point (the "then explore" half).
        if (self._goal_xy is not None and not self._goal_reached and not done
                and self._prev_goal_dist is not None):
            px, py = raw["position_x"], raw["position_y"]
            gx, gy = self._goal_xy
            gdist = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
            progress = self._prev_goal_dist - gdist
            if progress > 0:
                shaped += self._goal_scale * progress * explore_scale  # frontier pull
            self._prev_goal_dist = gdist
            if gdist <= self._goal_radius:
                self._goal_reached = True  # arrived: hand off to the exploration bonuses
        attacked = self._action_attacks[int(action)]  # this action pressed ATTACK
        if attacked and deltas["hitcount"] == 0 and not done:
            shaped -= self._miss_penalty * combat_scale  # don't punish blind shots while exploring

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
                # Memorise exit position — future episodes get proximity shaping toward it.
                if self._last_vars:
                    ex = self._last_vars["position_x"]
                    ey = self._last_vars["position_y"]
                    self._exit_pos = (ex, ey)
                    # Persist so other envs / the eval process / the next iteration inherit it.
                    if self._exit_store is not None:
                        try:
                            self._exit_store.save(self._current_map, ex, ey)
                        except OSError:
                            pass  # persistence must never break a finished episode
            elif cleared:
                shaped += 100.0               # cleared the enemies

        doom = {
            "deltas": deltas,
            "levels": levels,
            "action": int(action),
            "attacked": bool(attacked),  # tracker counts attacks from this (combined actions)
            "success": success,
        }
        if self._use_labels:
            doom["enemies_in_view"] = int(self._enemies_in_view)  # ground-truth on-screen count
            if self._combat_explore_split:
                doom["mode"] = "combat" if self._enemies_in_view > 0 else "explore"
        if done:
            doom["terminal"] = (
                "death" if dead else ("exit" if reached_exit else "timeout")
            )
            # Dense exit_progress (fairer than the binary exit_rate): how close, as a fraction
            # of the spawn→exit distance, the agent got to the KNOWN exit this episode. 1.0 =
            # reached it. Only defined once the exit has been found at least once on this map
            # (so the position is known); None otherwise.
            if reached_exit:
                doom["exit_progress"] = 1.0
            elif self._spawn_exit_dist and self._closest_exit_dist is not None:
                frac = 1.0 - (self._closest_exit_dist / self._spawn_exit_dist)
                doom["exit_progress"] = float(max(0.0, min(1.0, frac)))
            doom["base_return"] = self._ep_base  # native episode return (no shaping)
            doom["coverage_cells"] = len(self._visited)  # for frontier curriculum
            # The actual cells visited this episode -> persistent per-map heatmap memory.
            # Once per episode (off the hot path), so within the ±2% budget.
            doom["visited_cells"] = [[gx, gy] for (gx, gy) in self._visited]
            # Who killed the agent: the nearest monster at the moment of death (proxy).
            if dead and self._nearest_enemy:
                self._ep_enemies.setdefault(self._nearest_enemy, {})["killed_agent"] = 1
            doom["enemies"] = self._ep_enemies  # per-monster facts -> the bestiary
            # Episodic context for richer memory records.
            doom["nearest_enemy"] = self._nearest_enemy or ""
            px = self._last_vars.get("position_x", 0.0) if self._last_vars else 0.0
            py = self._last_vars.get("position_y", 0.0) if self._last_vars else 0.0
            doom["final_pos"] = [round(px), round(py)]
        if self._walls_pending:
            doom["walls"] = read_wall_segments(self.game)
            self._walls_pending = False
        return obs, float(shaped), done, False, {"map": self._current_map, "doom": doom}

    def close(self) -> None:
        self.game.close()


def make_campaign_env(
    cfg,
    doom_map: str,
    rank: int = 0,
    *,
    rewards: Optional[Dict[str, float]] = None,
    window_visible: Optional[bool] = None,
    memory_dir: Optional[str] = None,
):
    """Factory for SubprocVecEnv/DummyVecEnv.

    Takes the Config object and reads every perception/action flag from it, so adding a new
    one (depth, automap, strafe, ...) only touches CampaignDoomEnv + Config — not this
    signature and not the call sites. `rewards`/`window_visible`/`memory_dir` can override
    cfg for special cases (eval passes its own, BC passes none).
    """
    rewards = cfg.reward_weights() if rewards is None else rewards
    window_visible = cfg.render if window_visible is None else window_visible

    def _init():
        env = CampaignDoomEnv(
            wad_path=cfg.wad_path,
            doom_map=doom_map,
            frame_skip=cfg.frame_skip,
            resolution=cfg.resolution,
            episode_timeout=cfg.episode_timeout,
            kills_to_clear=cfg.kills_to_clear,
            window_visible=window_visible,
            rewards=rewards,
            spatial_memory=cfg.spatial_memory,
            memory_dir=memory_dir,
            depth_perception=cfg.depth_perception,
            strafe=cfg.strafe,
            automap=cfg.automap,
            use_labels=cfg.use_labels,
            game_vars=getattr(cfg, "game_vars", False),
        )
        env.reset(seed=cfg.seed + rank)
        return env

    return _init


def campaign_metadata(wad_path: str, doom_map: str, strafe: bool = False) -> Dict[str, Any]:
    """Discover button names / action count without booting training. `strafe` must match
    the training config so the discovered action count (-> brain name) is correct."""
    env = CampaignDoomEnv(wad_path=wad_path, doom_map=doom_map, strafe=strafe)
    meta = {"button_names": list(env.button_names), "num_actions": int(env.action_space.n)}
    env.close()
    return meta
