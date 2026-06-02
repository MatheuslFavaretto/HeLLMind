"""Accumulates per-monster encounter facts and persists the bestiary at training end.

On each finished episode it folds that episode's `doom["enemies"]` (from the campaign env)
plus the outcome and map into an in-memory accumulator — cheap, off the PPO hot path. It
touches disk once, in `on_training_end`, merging into the cross-run `BestiaryStore`.
"""
from collections import defaultdict
from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback

from writer.bestiary import BestiaryStore


class EnemyMemoryCallback(BaseCallback):
    def __init__(self, store: BestiaryStore, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.store = store
        self._acc: Dict[str, Dict[str, Any]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("episode") is None:
                continue
            doom = info.get("doom") or {}
            enemies = doom.get("enemies") or {}
            if not enemies:
                continue
            outcome = doom.get("terminal", "")
            map_name = info.get("map", "") or doom.get("map", "")
            for name, e in enemies.items():
                a = self._acc.setdefault(name, {
                    "encounters": 0, "seen": 0, "approach": 0, "killed": 0,
                    "killed_agent": 0, "total": 0, "ranged": False, "dist_min": 1e9,
                    "kill_weapon": defaultdict(int), "outcomes": defaultdict(int),
                    "maps": defaultdict(int),
                })
                a["encounters"] += 1
                a["seen"] += int(e.get("seen", 0))
                a["approach"] += int(e.get("approach", 0))
                a["killed"] += int(e.get("killed", 0))
                a["killed_agent"] += int(e.get("killed_agent", 0))
                a["total"] = max(a["total"], int(e.get("total", 0)))
                a["ranged"] = a["ranged"] or bool(e.get("ranged", False))
                a["dist_min"] = min(a["dist_min"], float(e.get("dist_min", 1e9)))
                for slot, n in (e.get("kill_weapon", {}) or {}).items():
                    a["kill_weapon"][str(slot)] += int(n)
                if outcome:
                    a["outcomes"][outcome] += 1
                if map_name:
                    a["maps"][map_name] += 1
        return True

    def on_training_end(self) -> None:
        if not self._acc:
            return
        # defaultdicts -> plain dicts for JSON.
        run = {n: {**a, "kill_weapon": dict(a["kill_weapon"]),
                   "outcomes": dict(a["outcomes"]), "maps": dict(a["maps"])}
               for n, a in self._acc.items()}
        try:
            self.store.merge(run)
        except Exception:
            pass  # persistence must never break a finished run
