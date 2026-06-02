"""Per-map explored heatmap that PERSISTS across runs (autonomy memory).

Each run discovers more of a level. This store accumulates, per map, how many times the
agent has stepped on each grid cell — summed over *every* run against the same vault — so
the agent (and the minimap) can "remember" the layout it has explored historically, not
just within one window.

Layout (default `<vault>/.memory/coverage/<MAP>.json`):
    {"map", "cell", "runs", "updated", "walls": [...], "cells": {"gx,gy": visits}}

It is written ONCE per run (merged at training end), never inside the PPO loop.
"""
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple


def _safe_map(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "UNKNOWN"))


class CoverageStore:
    def __init__(self, memory_dir: str) -> None:
        self.dir = os.path.join(memory_dir, "coverage")

    def _path(self, map_name: str) -> str:
        return os.path.join(self.dir, f"{_safe_map(map_name)}.json")

    # ------------------------------------------------------------------
    def load(self, map_name: str) -> Optional[Dict]:
        """Return the persisted record for a map, or None if nothing stored yet."""
        path = self._path(map_name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def load_cells(self, map_name: str) -> List[List[float]]:
        """Persisted cells as [gx, gy, visits] (the minimap's memory overlay)."""
        rec = self.load(map_name)
        if not rec:
            return []
        out: List[List[float]] = []
        for key, c in rec.get("cells", {}).items():
            gx, gy = key.split(",")
            out.append([float(gx), float(gy), float(c)])
        return out

    # ------------------------------------------------------------------
    def merge(
        self,
        map_name: str,
        cells: Dict[Tuple[int, int], int],
        walls: Optional[Sequence[Sequence[float]]] = None,
        cell_size: float = 96.0,
    ) -> Dict:
        """Add this run's visit counts into the map's persistent grid and write it back.
        `cells` maps (gx, gy) -> visits. Returns the merged record."""
        rec = self.load(map_name) or {
            "map": map_name, "cell": float(cell_size), "runs": 0,
            "walls": [], "cells": {},
        }
        merged: Counter = Counter()
        for key, c in rec.get("cells", {}).items():
            merged[key] += int(c)
        for (gx, gy), c in cells.items():
            merged[f"{int(gx)},{int(gy)}"] += int(c)
        rec["cells"] = dict(merged)
        rec["runs"] = int(rec.get("runs", 0)) + 1
        rec["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if walls:  # keep the latest geometry we saw (static per map)
            rec["walls"] = [list(w) for w in walls if w and len(w) == 4]
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._path(map_name) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, self._path(map_name))  # atomic: never leave a half file
        return rec
