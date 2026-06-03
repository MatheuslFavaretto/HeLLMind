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
from collections import OrderedDict
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
    def merge(self, map_name: str, positions: List[Tuple[float, float]]) -> Dict:
        """Fold this run's reached world positions into the archive (bucketed by cell)."""
        rec = self.load(map_name)
        cells = rec.get("cells", {})
        for (x, y) in positions:
            gx = round(x / self.cell_size)
            gy = round(y / self.cell_size)
            key = f"{gx},{gy}"
            if key in cells:
                cells[key]["visits"] = int(cells[key].get("visits", 0)) + 1
            else:
                cells[key] = {"x": float(x), "y": float(y), "visits": 1}
        rec["cells"] = cells
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._path(map_name) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, self._path(map_name))
        return rec

    # ------------------------------------------------------------------
    def sample_goal(
        self,
        map_name: str,
        spawn_xy: Tuple[float, float],
        rng: Optional[random.Random] = None,
        min_dist: float = 200.0,
    ) -> Optional[Tuple[float, float]]:
        """Pick a frontier cell to return to. Weight ∝ distance_from_spawn / (1 + visits),
        so the agent is sent toward FAR, RARELY-seen cells — the actual frontier, not the
        well-trodden area around spawn. Returns None if the archive has nothing far yet."""
        rng = rng or random
        sx, sy = spawn_xy
        weighted: List[Tuple[float, Tuple[float, float]]] = []
        for (x, y, visits) in self.cells(map_name):
            d = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if d < min_dist:
                continue  # too close to spawn to be a useful "return" target
            w = d / (1.0 + visits)
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
