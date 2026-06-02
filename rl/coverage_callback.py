"""Accumulates the per-map explored heatmap and persists it once, at training end.

During the run it just sums visited cells per map IN MEMORY (a cheap Counter update on
episode end). It touches disk a single time, in `on_training_end`, merging into the
cross-run `CoverageStore`. So the agent remembers the layouts it has explored across
runs, with no measurable training-time cost.
"""
from collections import Counter, defaultdict
from typing import Dict

from stable_baselines3.common.callbacks import BaseCallback

from writer.coverage_store import CoverageStore


class CoverageMemoryCallback(BaseCallback):
    def __init__(self, store: CoverageStore, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.store = store
        self._cells: Dict[str, Counter] = defaultdict(Counter)
        self._walls: Dict[str, list] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            doom = info.get("doom") or {}
            map_name = info.get("map", "") or doom.get("map", "")
            # Geometry is emitted once per map; keep it for the persisted overlay.
            walls = doom.get("walls")
            if walls and map_name:
                self._walls[map_name] = walls
            if info.get("episode") is None:
                continue  # only aggregate the visited grid on episode end
            for cell in doom.get("visited_cells", []) or []:
                try:
                    self._cells[map_name][(int(cell[0]), int(cell[1]))] += 1
                except (TypeError, ValueError, IndexError):
                    continue
        return True

    def on_training_end(self) -> None:
        for map_name, counter in self._cells.items():
            if not map_name or not counter:
                continue
            try:
                self.store.merge(map_name, dict(counter), walls=self._walls.get(map_name))
            except Exception:
                pass  # persistence must never break a finished training run
