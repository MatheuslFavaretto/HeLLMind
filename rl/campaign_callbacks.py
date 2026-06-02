"""Map curriculum: advances sequentially to the next map by timesteps.

The agent trains `steps_per_map` on each map and then moves to the next in the list.
When the list ends, it repeats (if loop) or stops switching. The switch is applied via
`env_method("set_map", ...)`; each env applies it on reset.
"""
from collections import Counter
from typing import Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback


def map_step_weights(events: List[dict], maps: List[str]) -> Dict[str, float]:
    """Closed-loop feedback: weight each map by how often the agent DIED there (from
    persistent memory), normalized so the mean is 1.0. More deaths -> more training.
    With no memory it returns all 1.0 (uniform = the previous behavior)."""
    deaths = Counter(
        e.get("map") for e in events
        if e.get("type") == "death" and e.get("map")
    )
    raw = {m: float(deaths.get(m, 0)) + 1.0 for m in maps}  # +1 smoothing
    mean = sum(raw.values()) / len(maps)
    return {m: raw[m] / mean for m in maps}


class MapCurriculumCallback(BaseCallback):
    def __init__(
        self,
        maps: List[str],
        steps_per_map: int,
        loop_maps: bool = False,
        weights: Optional[Dict[str, float]] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.maps = maps
        self.steps_per_map = steps_per_map
        self.loop_maps = loop_maps
        # Per-map step budgets: memory-weighted (mean 1.0) so harder maps get more steps.
        w = weights or {m: 1.0 for m in maps}
        self.budgets = [max(1, int(steps_per_map * w.get(m, 1.0))) for m in maps]
        self._map_idx = 0
        self._next_switch = self.budgets[0]  # first switch after map 0's budget

    def _current_map(self) -> str:
        return self.maps[self._map_idx]

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_switch:
            return True

        # Time to advance to the next map.
        last = self._map_idx >= len(self.maps) - 1
        if last and not self.loop_maps:
            # Already on the last map and no loop: nothing to switch.
            self._next_switch = float("inf")
            if self.verbose:
                print(f"[curriculum] last map ({self._current_map()}) — no more switches.")
            return True

        self._map_idx = (self._map_idx + 1) % len(self.maps)
        new_map = self._current_map()
        # Schedule the new map on all envs (applied on each one's next reset).
        self.training_env.env_method("set_map", new_map)
        self._next_switch = self.num_timesteps + self.budgets[self._map_idx]
        if self.verbose:
            print(
                f"[curriculum] step={self.num_timesteps} -> switching to {new_map} "
                f"({self._map_idx + 1}/{len(self.maps)})"
            )
        return True
