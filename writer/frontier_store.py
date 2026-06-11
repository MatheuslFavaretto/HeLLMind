"""Per-map frontier archive for Go-Explore-style "return, then explore" (autonomy memory).

Go-Explore (Ecoffet et al., Nature 2021) breaks hard-exploration by *remembering promising
states, returning to them, and exploring from there* — instead of always restarting from
the start state and hoping to wander far enough. ViZDoom can't teleport the agent, so we do
the tractable equivalent: archive the world positions the agent has reached (bucketed into
coarse cells), and at episode start sometimes hand it a far, rarely-seen cell as a GOAL.
A dense potential reward guides it *back* to that frontier; once there, the normal
exploration bonuses take over and it pushes outward from a fresh launch point — exactly the
"return then explore" idea, without save-states or teleport.

Layout (default `<vault>/.memory/frontier/<MAP>.json`):
    {"map", "cell", "cells": {"gx,gy": {"x", "y", "visits"}}}

Merged once per run (off the hot path).
"""
import json
import os
import random
from typing import Dict, List, Optional, Tuple


def _safe_map(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "UNKNOWN"))


class FrontierStore:
    def __init__(self, memory_dir: str, cell_size: float = 96.0) -> None:
        self.dir = os.path.join(memory_dir, "frontier")
        self.cell_size = float(cell_size)

    def _path(self, map_name: str) -> str:
        return os.path.join(self.dir, f"{_safe_map(map_name)}.json")

    # ------------------------------------------------------------------
    def load(self, map_name: str) -> Dict:
        path = self._path(map_name)
        if not os.path.exists(path):
            return {"map": map_name, "cell": self.cell_size, "cells": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"map": map_name, "cell": self.cell_size, "cells": {}}

    def cells(self, map_name: str) -> List[Tuple[float, float, int]]:
        """Archived frontier points as (x, y, visits)."""
        rec = self.load(map_name)
        out: List[Tuple[float, float, int]] = []
        for c in rec.get("cells", {}).values():
            try:
                out.append((float(c["x"]), float(c["y"]), int(c.get("visits", 1))))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    # ------------------------------------------------------------------
    def merge(self, map_name: str, positions: List[Tuple[float, float]],
              max_age: int = 25) -> Dict:
        """Fold this run's reached positions into the archive (bucketed by cell).

        Frontier AGING: each merge bumps a generation counter and stamps every touched cell
        with the current generation. Cells not revisited for `max_age` generations are pruned
        — so frontiers that turned out to be dead ends fade instead of being chased forever.
        """
        rec = self.load(map_name)
        cells = rec.get("cells", {})
        gen = int(rec.get("gen", 0)) + 1
        for (x, y) in positions:
            gx = round(x / self.cell_size)
            gy = round(y / self.cell_size)
            key = f"{gx},{gy}"
            if key in cells:
                cells[key]["visits"] = int(cells[key].get("visits", 0)) + 1
            else:
                cells[key] = {"x": float(x), "y": float(y), "visits": 1}
            cells[key]["last_gen"] = gen  # freshness stamp (aging)
        # Prune stale cells (aged out) to keep the archive a live frontier, not a junk pile.
        cells = {k: c for k, c in cells.items()
                 if gen - int(c.get("last_gen", 0)) <= max_age}
        rec["cells"] = cells
        rec["gen"] = gen
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._path(map_name) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, self._path(map_name))
        return rec

    # ------------------------------------------------------------------
    _AGING_DECAY = 0.9  # per-generation weight decay for cells not seen recently

    def sample_goal(
        self,
        map_name: str,
        spawn_xy: Tuple[float, float],
        rng: Optional[random.Random] = None,
        min_dist: float = 200.0,
        route_dist_fn=None,
    ) -> Optional[Tuple[float, float]]:
        """Pick a frontier cell to return to, weighting three signals (Go-Explore + frontier
        intelligence):
          • depth / (1+visits) — far + rarely-seen. Depth is GEODESIC route depth when
            `route_dist_fn(x, y) -> dist_to_exit | None` is given (return-to-the-deepest-
            point-ON-THE-ROUTE — the door-consolidation mechanism), euclidean-from-spawn
            otherwise. Off-route cells (fn returns None) are SKIPPED entirely: the dive-era
            archive contains pit cells, and sending the agent back into an inescapable pit
            as a 'goal' would re-teach the exploit the reward fix just sealed.
          • EDGE bonus 1/(1+neighbours) — cells on the boundary of the explored region.
          • AGING decay 0.9^(gen-last_gen) — fade frontiers that haven't paid off lately.
        Returns None if the archive has nothing far yet."""
        rng = rng or random
        sx, sy = spawn_xy
        rec = self.load(map_name)
        cells = rec.get("cells", {})
        gen = int(rec.get("gen", 0))
        keys = set(cells.keys())
        spawn_route = route_dist_fn(sx, sy) if route_dist_fn else None
        weighted: List[Tuple[float, Tuple[float, float]]] = []
        for key, c in cells.items():
            try:
                x, y = float(c["x"]), float(c["y"])
                visits = int(c.get("visits", 1))
            except (KeyError, TypeError, ValueError):
                continue
            d = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if d < min_dist:
                continue  # too close to spawn to be a useful "return" target
            if route_dist_fn is not None:
                cell_route = route_dist_fn(x, y)
                if cell_route is None:
                    continue  # off the walkable route (pit etc.) — never a goal
                if spawn_route is not None:
                    d = max(0.0, spawn_route - cell_route)  # depth ALONG the route
                    if d <= 0:
                        continue
            neighbours = self._neighbour_count(key, keys)
            edge = 1.0 / (1.0 + neighbours)                      # prioritise the boundary
            age = gen - int(c.get("last_gen", gen))
            decay = self._AGING_DECAY ** max(0, age)             # fade stale frontiers
            w = (d / (1.0 + visits)) * edge * decay
            if w > 0:
                weighted.append((w, (x, y)))
        if not weighted:
            return None
        total = sum(w for w, _ in weighted)
        if total <= 0:
            return None
        r = rng.uniform(0, total)
        acc = 0.0
        for w, pos in weighted:
            acc += w
            if r <= acc:
                return pos
        return weighted[-1][1]

    @staticmethod
    def _neighbour_count(key: str, keys: set) -> int:
        """How many of the 8 surrounding cells are also archived (interior-ness)."""
        try:
            gx, gy = (int(v) for v in key.split(","))
        except ValueError:
            return 0
        return sum(
            1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and f"{gx + dx},{gy + dy}" in keys
        )
