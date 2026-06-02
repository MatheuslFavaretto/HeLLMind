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

    # ------------------------------------------------------------------
    def update(self, infos: List[dict], actions: np.ndarray) -> None:
        """Called every vec-env step, for all parallel envs."""
        for info, act in zip(infos, actions):
            doom = info.get("doom")
            if doom is not None:
                for k, v in doom["deltas"].items():
                    self.delta_sums[k] = self.delta_sums.get(k, 0.0) + float(v)
                for n in LEVELS:
                    self.level_samples[n].append(doom["levels"][n])
                self.action_counts[int(doom["action"])] += 1
                # Coverage/path: discretize position into a grid and count visits.
                px = doom["levels"].get("position_x", 0.0)
                py = doom["levels"].get("position_y", 0.0)
                self.cell_counts[
                    (round(px / COVERAGE_CELL), round(py / COVERAGE_CELL))
                ] += 1
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

    # ------------------------------------------------------------------
    def snapshot(self, num_timesteps: int) -> Dict[str, Any]:
        """Summary of the current window. Everything here becomes LLM context."""
        n_eps = len(self.episode_rewards)
        d = self.delta_sums
        shots = float(d.get("hitcount", 0.0))  # shots that landed
        # 'attack' is usually the fire button; we compute hits per attack.
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
            # aim (hits vs misses)
            "shots_fired": float(attack_count),
            "shots_hit": shots,
            "shots_missed": misses,
            "shooting_accuracy": float(accuracy),
            "kills_per_episode": (d.get("killcount", 0.0) / n_eps) if n_eps else 0.0,
            # exploration / path taken
            "distance_traveled": d.get("distance", 0.0),
            "distance_per_episode": (d.get("distance", 0.0) / n_eps) if n_eps else 0.0,
            "cells_visited": len(self.cell_counts),
            "map_coverage": self._coverage(),
            "weapons_used": self._weapon_distribution(),
            # cells [gx, gy, visits] to render the path minimap
            "path_cells": [[gx, gy, c] for (gx, gy), c in self.cell_counts.items()],
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
    def _coverage(self) -> Dict[str, float]:
        """Estimate the explored fraction: visited cells / bounding-box area.

        We don't know the real map size, so we normalize by the bounding box of the
        seen positions — an honest approximation of 'how much of the traversed area
        the agent actually covered'.
        """
        xs = self.level_samples.get("position_x", [])
        ys = self.level_samples.get("position_y", [])
        n_cells = len(self.cell_counts)
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
        }

    def _weapon_distribution(self) -> Dict[str, float]:
        """Fraction of time with each selected weapon (ViZDoom slot)."""
        samples = self.level_samples.get("selected_weapon", [])
        if not samples:
            return {}
        counts = Counter(int(s) for s in samples)
        total = float(sum(counts.values()))
        return {f"slot_{k}": v / total for k, v in sorted(counts.items())}
