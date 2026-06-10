"""Accumulates ViZDoom signal over a window and produces a summarized snapshot.

Philosophy: report DELTAS (what changed since the last note), not totals, so the
notes don't all become 'the numbers went up again'.
"""
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from instrumentation.action_stats import (
    action_distribution,
    action_entropy,
    max_entropy,
)
from instrumentation.game_vars import LEVELS

# Cell size (in map units) when discretizing position to estimate map COVERAGE
# (how many distinct cells the agent stepped on).
COVERAGE_CELL = 96.0


class StatsTracker:
    def __init__(self, button_names: List[str]) -> None:
        self.button_names = button_names
        self.n_actions = len(button_names)
        # Map geometry (walls) — static; persists across windows.
        self.map_walls: list = []
        self.reset_window()

    def reset_window(self) -> None:
        # Dynamic sum: accepts any delta key coming from the env
        # (killcount, hitcount, ..., and also "distance").
        self.delta_sums: Dict[str, float] = {}
        self.level_samples: Dict[str, List[float]] = {n: [] for n in LEVELS}
        self.action_counts = np.zeros(self.n_actions, dtype=np.float64)
        # Attack accounting. With COMBINED actions (campaign) several actions press ATTACK,
        # so we can't infer "did it shoot?" from a single button index — the env reports it
        # per step via doom["attacked"]. When that's present we count it directly; otherwise
        # we fall back to the single-button method (scenario env).
        self.attack_steps = 0
        self._has_attack_flag = False
        # Combat/exploration split telemetry (needs USE_LABELS): how the agent spends its
        # time and whether it actually FIGHTS when it sees an enemy — the two regimes the
        # reward decoupling separates, so the coach can tune each one independently.
        self.combat_steps = 0          # steps with an enemy on screen
        self.explore_steps = 0         # steps with a clear screen
        self.combat_attack_steps = 0   # of combat steps, how many it pressed ATTACK
        self.combat_hits = 0.0         # hits landed while in combat
        self._has_mode_flag = False
        self.episode_rewards: List[float] = []
        self.base_returns: List[float] = []  # native (unshaped) episode returns
        self.episode_lengths: List[int] = []
        self.steps_in_window = 0
        # Coverage/path: how many times the agent stepped on each cell (heatmap).
        self.cell_counts: Counter = Counter()
        self.attack_actions = 0  # number of attack actions (for shooting accuracy)
        # Campaign: current map and count of "completed" episodes.
        self.current_map: str = ""
        self.episodes_done = 0
        self.episodes_success = 0
        # How episodes ended (death / exit / timeout) — "exit" = reached the level end.
        self.terminals: Counter = Counter()
        self.coverage_cells_per_ep: List[int] = []  # distinct cells per episode
        self.exit_progress_per_ep: List[float] = []  # how close to the exit (dense, fairer)
        self.route_progress_per_ep: List[float] = []  # geodesic: fraction of the REAL route
        self.death_route_dist_per_ep: List[float] = []  # geodesic dist-to-exit at death
        self.enemies_seen_per_ep: List[int] = []     # distinct enemies seen per episode
        # Richer telemetry (aim / movement / weapons / perception).
        self.nearest_centered_samples: List[float] = []  # aim quality: enemy off-centre [0,1]
        self.idle_steps = 0                          # steps with ~no movement (stuck/idle)
        self.seen_counts_per_ep: List[Dict[str, int]] = []  # distinct objects seen, by category
        self.doors_reached_per_ep: List[int] = []    # doors opened/reached per episode
        self.reward_parts_sum: Dict[str, float] = {}  # reward breakdown by component (summed)
        self.frontier_reach_per_ep: List[float] = []  # farthest from spawn per episode
        self.reaction_ticks_per_ep: List[float] = []  # mean enemy-seen→first-shot per episode
        self.best_weapon_frac_per_ep: List[float] = []  # share wielding strongest owned gun
        self.weapon_shots: Counter = Counter()       # shots fired per weapon slot
        self.weapon_hits: Counter = Counter()        # hits landed per weapon slot
        self.nearest_enemy_samples: List[float] = []  # distance to nearest enemy (positioning)
        # Ordered trajectory of ONE representative env (env 0) so the minimap can draw
        # the path as a CONNECTED LINE (visit order), not just an unordered heatmap.
        # The heatmap mixes all parallel envs; an ordered line only makes sense per env.
        self._env0_path: List[tuple] = []        # cells of env 0's CURRENT episode
        self._env0_last_path: List[tuple] = []   # last COMPLETED episode (the one we draw)

    # ------------------------------------------------------------------
    def update(self, infos: List[dict], actions: np.ndarray) -> None:
        """Called every vec-env step, for all parallel envs."""
        for idx, (info, act) in enumerate(zip(infos, actions)):
            doom = info.get("doom")
            if doom is not None:
                for k, v in doom["deltas"].items():
                    self.delta_sums[k] = self.delta_sums.get(k, 0.0) + float(v)
                # Idle/stuck: a step that barely moved (movement-quality signal).
                if float(doom["deltas"].get("distance", 1.0)) < 4.0:
                    self.idle_steps += 1
                # Aim quality: how off-centre the nearest enemy was (only when one was visible).
                if "nearest_centered" in doom:
                    self.nearest_centered_samples.append(float(doom["nearest_centered"]))
                if "nearest_enemy_dist" in doom:
                    self.nearest_enemy_samples.append(float(doom["nearest_enemy_dist"]))
                # Reward breakdown: what's driving the agent (combat/explore/engage/...).
                for k, v in (doom.get("reward_parts") or {}).items():
                    self.reward_parts_sum[k] = self.reward_parts_sum.get(k, 0.0) + float(v)
                # Accuracy by weapon: shots + hits keyed by the wielded slot.
                if doom.get("attacked"):
                    slot = int(doom["levels"].get("selected_weapon", 0))
                    self.weapon_shots[slot] += 1
                    self.weapon_hits[slot] += float(doom["deltas"].get("hitcount", 0.0))
                for n in LEVELS:
                    if n in doom["levels"]:  # tolerate vars a synthetic/old info omits
                        self.level_samples[n].append(doom["levels"][n])
                self.action_counts[int(doom["action"])] += 1
                if "attacked" in doom:
                    self._has_attack_flag = True
                    if doom["attacked"]:
                        self.attack_steps += 1
                # Combat vs exploration regime (ground-truth enemy visibility).
                if "enemies_in_view" in doom:
                    self._has_mode_flag = True
                    in_combat = int(doom["enemies_in_view"]) > 0
                    if in_combat:
                        self.combat_steps += 1
                        if doom.get("attacked"):
                            self.combat_attack_steps += 1
                        self.combat_hits += float(doom["deltas"].get("hitcount", 0.0))
                    else:
                        self.explore_steps += 1
                # Coverage/path: discretize position into a grid and count visits.
                px = doom["levels"].get("position_x", 0.0)
                py = doom["levels"].get("position_y", 0.0)
                cell = (round(px / COVERAGE_CELL), round(py / COVERAGE_CELL))
                self.cell_counts[cell] += 1
                # Ordered trajectory of env 0 only (for the connected-line minimap),
                # de-duplicating consecutive repeats so the line reflects movement.
                if idx == 0 and (not self._env0_path or self._env0_path[-1] != cell):
                    self._env0_path.append(cell)
                walls = doom.get("walls")
                if walls:  # sent once per map; we keep it for the minimap
                    self.map_walls = walls
            if info.get("map"):
                self.current_map = info["map"]
            self.steps_in_window += 1

            ep = info.get("episode")
            if ep is not None:  # Monitor finished an episode
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(int(ep["l"]))
                self.episodes_done += 1
                if doom is not None:
                    if doom.get("success"):
                        self.episodes_success += 1
                    if "base_return" in doom:
                        self.base_returns.append(float(doom["base_return"]))
                    if doom.get("terminal"):
                        self.terminals[doom["terminal"]] += 1
                    if "coverage_cells" in doom:
                        self.coverage_cells_per_ep.append(int(doom["coverage_cells"]))
                    if "exit_progress" in doom:
                        self.exit_progress_per_ep.append(float(doom["exit_progress"]))
                    if "route_progress" in doom:
                        self.route_progress_per_ep.append(float(doom["route_progress"]))
                    if "death_route_dist" in doom:
                        self.death_route_dist_per_ep.append(float(doom["death_route_dist"]))
                    if "enemies_seen" in doom:
                        self.enemies_seen_per_ep.append(int(doom["enemies_seen"]))
                    if "seen_counts" in doom:
                        self.seen_counts_per_ep.append(dict(doom["seen_counts"]))
                    if "doors_reached" in doom:
                        self.doors_reached_per_ep.append(int(doom["doors_reached"]))
                    if "frontier_reach" in doom:
                        self.frontier_reach_per_ep.append(float(doom["frontier_reach"]))
                    if "reaction_ticks" in doom:
                        self.reaction_ticks_per_ep.append(float(doom["reaction_ticks"]))
                    if "best_weapon_fraction" in doom:
                        self.best_weapon_frac_per_ep.append(float(doom["best_weapon_fraction"]))
                # env 0 finished an episode: keep its (ordered) path as the one to draw,
                # then start a fresh trajectory for the next episode.
                if idx == 0 and len(self._env0_path) >= 2:
                    self._env0_last_path = self._env0_path
                    self._env0_path = []

    # ------------------------------------------------------------------
    def snapshot(self, num_timesteps: int) -> Dict[str, Any]:
        """Summary of the current window. Everything here becomes LLM context."""
        n_eps = len(self.episode_rewards)
        d = self.delta_sums
        shots = float(d.get("hitcount", 0.0))  # shots that landed
        # Prefer the env-reported attack count (handles combined actions); else infer it
        # from the single ATTACK button (scenario env, one-hot actions).
        if self._has_attack_flag:
            attack_count = float(self.attack_steps)
        else:
            attack_idx = next(
                (i for i, n in enumerate(self.button_names) if "ATTACK" in n.upper()),
                None,
            )
            attack_count = (
                float(self.action_counts[attack_idx]) if attack_idx is not None else 0.0
            )
        accuracy = (shots / attack_count) if attack_count > 0 else 0.0
        misses = max(0.0, attack_count - shots)

        def _mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        def _min(xs):
            return float(np.min(xs)) if xs else 0.0

        snap = {
            "num_timesteps": int(num_timesteps),
            "steps_in_window": int(self.steps_in_window),
            "episodes": n_eps,
            "map": self.current_map,
            "success_rate": (
                self.episodes_success / self.episodes_done
                if self.episodes_done
                else 0.0
            ),
            # Completion: fraction of episodes that ended by reaching the level EXIT.
            "exit_rate": (
                self.terminals.get("exit", 0) / self.episodes_done
                if self.episodes_done
                else 0.0
            ),
            "terminals": dict(self.terminals),  # {death, exit, timeout} counts
            # Dense "how close to the exit" (mean fraction, 1.0 = reached) — fairer than the
            # binary exit_rate. Only populated once the exit position is known on the map.
            "exit_progress": _mean(self.exit_progress_per_ep),
            "route_progress": _mean(self.route_progress_per_ep),
            # Best single episode: the route FRONTIER (mean hides one breakthrough run).
            "route_progress_best": max(self.route_progress_per_ep, default=0.0),
            # Mean geodesic distance-to-exit at death: ~3650 = dying in the spawn pocket,
            # ~2240 = dying in the gauntlet (= the agent is finally pushing the route).
            "death_route_dist": _mean(self.death_route_dist_per_ep),
            "mean_reward": _mean(self.episode_rewards),
            "mean_base_reward": _mean(self.base_returns),  # native, shaping-independent
            "mean_episode_length": _mean(self.episode_lengths),
            "min_episode_length": _min(self.episode_lengths),
            "max_episode_length": float(max(self.episode_lengths)) if self.episode_lengths else 0.0,
            # counters (deltas over the window)
            "kills": d.get("killcount", 0.0),
            "hits_landed": d.get("hitcount", 0.0),
            "hits_taken": d.get("hits_taken", 0.0),
            "damage_dealt": d.get("damagecount", 0.0),
            "damage_taken": d.get("damage_taken", 0.0),
            "deaths": d.get("deathcount", 0.0),
            "items_collected": d.get("itemcount", 0.0),
            # "What happened this run" (per the user's preferred, non-binary metrics):
            "enemies_seen_total": float(sum(self.enemies_seen_per_ep)),
            "enemies_seen_per_episode": _mean(self.enemies_seen_per_ep),
            "heals_consumed": d.get("heal_pickups", 0.0),    # health pickups taken
            "health_recovered": d.get("heal_hp", 0.0),       # total HP regained from them
            "hits_taken_per_episode": (d.get("hits_taken", 0.0) / n_eps) if n_eps else 0.0,
            "shots_fired_per_episode": (float(attack_count) / n_eps) if n_eps else 0.0,
            # Combat efficiency: shots spent per kill (lower = better aim/ammo discipline) and
            # damage soaked per kill (lower = survives the fight better). inf-safe.
            "shots_per_kill": (float(attack_count) / d.get("killcount", 0.0))
            if d.get("killcount", 0.0) > 0 else 0.0,
            "damage_taken_per_kill": (d.get("damage_taken", 0.0) / d.get("killcount", 0.0))
            if d.get("killcount", 0.0) > 0 else 0.0,
            # AIM: how off-centre enemies were (lower=better aim), and the share of shots fired
            # with NO enemy on screen (spray discipline — high = tighten the trigger).
            "aim_offset": _mean(self.nearest_centered_samples),
            "wasted_shot_rate": ((self.attack_steps - self.combat_attack_steps)
                                 / self.attack_steps) if self.attack_steps else 0.0,
            # MOVEMENT: share of steps barely moving (stuck/idle).
            "idle_rate": (self.idle_steps / self.steps_in_window) if self.steps_in_window else 0.0,
            # WEAPONS: how many distinct weapons it wielded and how often it switched.
            **self._weapon_choice_metrics(n_eps),
            # PERCEPTION: distinct objects identified per category + doors reached, per episode.
            **self._perception_metrics(),
            # Conversion / survival / reward-breakdown / weapon-accuracy (the rest of the suite).
            **self._advanced_metrics(d, n_eps),
            # aim (hits vs misses)
            "shots_fired": float(attack_count),
            "shots_hit": shots,
            "shots_missed": misses,
            "shooting_accuracy": float(accuracy),
            "kills_per_episode": (d.get("killcount", 0.0) / n_eps) if n_eps else 0.0,
            # Combat vs exploration regime (needs USE_LABELS). combat_fraction = how much
            # time it spent facing enemies; combat_engagement = of that time, how often it
            # actually SHOT (low = the passivity the user saw); combat_accuracy = aim in
            # combat. These let the coach tune COMBAT and EXPLORATION separately.
            **self._mode_metrics(),
            # exploration / path taken
            "distance_traveled": d.get("distance", 0.0),
            "distance_per_episode": (d.get("distance", 0.0) / n_eps) if n_eps else 0.0,
            "cells_visited": len(self.cell_counts),
            "map_coverage": self._coverage(),
            "weapons_used": self._weapon_distribution(),
            # cells [gx, gy, visits] to render the path minimap (heatmap)
            "path_cells": [[gx, gy, c] for (gx, gy), c in self.cell_counts.items()],
            # ordered [gx, gy] of one episode -> minimap draws it as a connected line
            "path_polyline": [[gx, gy] for (gx, gy) in (self._env0_last_path or self._env0_path)],
            # real map walls [x1,y1,x2,y2] (minimap background)
            "map_walls": self.map_walls,
            # levels
            "mean_health": _mean(self.level_samples["health"]),
            "min_health": _min(self.level_samples["health"]),
            "mean_ammo": _mean(self.level_samples["ammo2"]),
            # actions
            "action_distribution": action_distribution(
                self.action_counts, self.button_names
            ),
            "action_entropy": action_entropy(self.action_counts),
            "action_entropy_normalized": (
                action_entropy(self.action_counts) / max_entropy(self.n_actions)
                if max_entropy(self.n_actions) > 0
                else 0.0
            ),
        }
        return snap

    # ------------------------------------------------------------------
    def _mode_metrics(self) -> Dict[str, float]:
        """Combat/exploration regime split (empty-ish if USE_LABELS is off)."""
        if not self._has_mode_flag:
            return {"combat_fraction": 0.0, "combat_engagement": 0.0, "combat_accuracy": 0.0}
        total = self.combat_steps + self.explore_steps
        return {
            # share of decision steps spent with an enemy on screen
            "combat_fraction": (self.combat_steps / total) if total else 0.0,
            # of combat steps, how often it pressed ATTACK (low = passive when it sees enemies)
            "combat_engagement": (self.combat_attack_steps / self.combat_steps)
            if self.combat_steps else 0.0,
            # aim while in combat (hits per attack step)
            "combat_accuracy": (self.combat_hits / self.combat_attack_steps)
            if self.combat_attack_steps else 0.0,
        }

    def _coverage(self) -> Dict[str, float]:
        """Explored fraction of the map.

        Preferred: visited cells / cells in the REAL map bounding box (from wall
        geometry) — a true 'how much of the level did it see'. Fallback (no walls):
        normalize by the bounding box of the seen positions.
        """
        n_cells = len(self.cell_counts)
        # True coverage: grid the real map extent (walls) and see how much we touched.
        if self.map_walls:
            xs_w = [c for w in self.map_walls for c in (w[0], w[2])]
            ys_w = [c for w in self.map_walls for c in (w[1], w[3])]
            gx = (max(xs_w) - min(xs_w)) / COVERAGE_CELL
            gy = (max(ys_w) - min(ys_w)) / COVERAGE_CELL
            map_cells = max(1.0, (gx + 1.0) * (gy + 1.0))
            return {
                "cells_visited": float(n_cells),
                "map_cells": float(round(map_cells, 1)),
                "explored_fraction": float(min(1.0, n_cells / map_cells)),
                "source": "walls",
            }
        xs = self.level_samples.get("position_x", [])
        ys = self.level_samples.get("position_y", [])
        if len(xs) < 2:
            return {"cells_visited": float(n_cells), "explored_fraction": 0.0}
        span_x = (max(xs) - min(xs)) / COVERAGE_CELL
        span_y = (max(ys) - min(ys)) / COVERAGE_CELL
        # Essentially stationary (e.g. defend_the_center turret): the agent didn't
        # move, so "% explored" is a meaningless artifact of a tiny bbox -> 0.0.
        if span_x < 1.0 and span_y < 1.0:
            return {"cells_visited": float(n_cells), "explored_fraction": 0.0,
                    "static": True}
        bbox_cells = max(1.0, (span_x + 1.0) * (span_y + 1.0))
        return {
            "cells_visited": float(n_cells),
            "bbox_cells": float(round(bbox_cells, 1)),
            "explored_fraction": float(min(1.0, n_cells / bbox_cells)),
            "source": "bbox",
        }

    def _advanced_metrics(self, d: Dict[str, float], n_eps: int) -> Dict[str, Any]:
        """Conversion rates, survival, reward breakdown, per-weapon accuracy, frontier/reaction."""
        out: Dict[str, Any] = {}
        # Conversion: of enemies/pickups SEEN, how many did it kill / collect?
        seen_enemies = float(sum(self.enemies_seen_per_ep))
        out["kill_conversion"] = (d.get("killcount", 0.0) / seen_enemies) if seen_enemies else 0.0
        pickups_seen = float(sum(c.get(k, 0) for c in self.seen_counts_per_ep
                                 for k in ("health", "ammo", "weapon", "key", "item")))
        out["pickup_conversion"] = (d.get("itemcount", 0.0) / pickups_seen) if pickups_seen else 0.0
        # Survival: time in the danger zone + out of ammo.
        hp = self.level_samples.get("health", [])
        out["low_health_fraction"] = (sum(1 for h in hp if h < 30) / len(hp)) if hp else 0.0
        am = self.level_samples.get("ammo2", [])
        out["out_of_ammo_fraction"] = (sum(1 for a in am if a <= 0) / len(am)) if am else 0.0
        if self.nearest_enemy_samples:   # mean distance to the nearest enemy (positioning)
            out["nearest_enemy_dist"] = float(np.mean(self.nearest_enemy_samples))
        # Movement: circling — share of position samples on an already-visited cell.
        total_visits = sum(self.cell_counts.values())
        out["revisit_rate"] = ((total_visits - len(self.cell_counts)) / total_visits
                               if total_visits else 0.0)
        # Frontier reach / reaction time (means over episodes).
        if self.frontier_reach_per_ep:
            out["frontier_reach"] = float(np.mean(self.frontier_reach_per_ep))
        if self.reaction_ticks_per_ep:
            out["reaction_ticks"] = float(np.mean(self.reaction_ticks_per_ep))
        if self.best_weapon_frac_per_ep:
            out["best_weapon_fraction"] = float(np.mean(self.best_weapon_frac_per_ep))
        # Reward breakdown as fractions of the total MAGNITUDE (so penalties show their weight).
        total_mag = sum(abs(v) for v in self.reward_parts_sum.values()) or 1.0
        out["reward_breakdown"] = {k: v / total_mag for k, v in self.reward_parts_sum.items()}
        # Accuracy per weapon slot (hits / shots).
        out["accuracy_by_weapon"] = {
            f"slot_{slot}": self.weapon_hits[slot] / self.weapon_shots[slot]
            for slot in sorted(self.weapon_shots) if self.weapon_shots[slot] > 0
        }
        return out

    def _weapon_choice_metrics(self, n_eps: int) -> Dict[str, float]:
        """WEAPONS: how many distinct slots it wielded + how often it switched (NEXTW use)."""
        s = self.level_samples.get("selected_weapon", [])
        if not s:
            return {"distinct_weapons_used": 0.0, "weapon_switches_per_episode": 0.0}
        switches = sum(1 for i in range(1, len(s)) if int(s[i]) != int(s[i - 1]))
        return {
            "distinct_weapons_used": float(len(set(int(x) for x in s))),
            "weapon_switches_per_episode": (switches / n_eps) if n_eps else 0.0,
        }

    def _perception_metrics(self) -> Dict[str, Any]:
        """PERCEPTION: distinct objects IDENTIFIED per category + doors reached, per episode."""
        out: Dict[str, Any] = {}
        if self.seen_counts_per_ep:
            cats = set().union(*[set(d) for d in self.seen_counts_per_ep])
            out["objects_seen_per_episode"] = {
                c: float(np.mean([d.get(c, 0) for d in self.seen_counts_per_ep])) for c in cats
            }
        if self.doors_reached_per_ep:
            out["doors_reached_per_episode"] = float(np.mean(self.doors_reached_per_ep))
        return out

    def _weapon_distribution(self) -> Dict[str, float]:
        """Fraction of time with each selected weapon (ViZDoom slot)."""
        samples = self.level_samples.get("selected_weapon", [])
        if not samples:
            return {}
        counts = Counter(int(s) for s in samples)
        total = float(sum(counts.values()))
        return {f"slot_{k}": v / total for k, v in sorted(counts.items())}
