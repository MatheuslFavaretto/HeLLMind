"""Per-map EXIT position that PERSISTS across runs (autonomy memory).

The level exit is a sparse, accidental discovery: the agent only ever sees EXIT_REWARD if
it happens to stumble onto the exit. That signal is worthless to the *next* episode/run
unless we remember WHERE the exit was. This store records, per map, the (x, y) of the exit
the first time it's reached — so every later episode can be shaped with a dense gradient
toward it (`exit_prox_scale` in campaign.py), turning a one-in-a-million stumble into a
reusable goal.

Layout (default `<vault>/.memory/exits/<MAP>.json`):
    {"map", "x", "y", "updated"}

Written ONCE, the first time the exit is reached on a map (cheap, off the hot path).
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional, Tuple


def _safe_map(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "UNKNOWN"))


class ExitStore:
    def __init__(self, memory_dir: str) -> None:
        self.dir = os.path.join(memory_dir, "exits")

    def _path(self, map_name: str) -> str:
        return os.path.join(self.dir, f"{_safe_map(map_name)}.json")

    def load(self, map_name: str) -> Optional[Tuple[float, float]]:
        """The memorised exit (x, y) for a map, or None if never reached yet."""
        path = self._path(map_name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            return float(rec["x"]), float(rec["y"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            return None

    def save(self, map_name: str, x: float, y: float) -> None:
        """Record the exit position (atomic). Idempotent — last writer wins, which is fine
        since a map's exit is static."""
        rec = {
            "map": map_name,
            "x": float(x),
            "y": float(y),
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._path(map_name) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, self._path(map_name))
