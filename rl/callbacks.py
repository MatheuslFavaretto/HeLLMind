"""Callback that collects metrics every step and SERIALIZES new snapshots.

IMPORTANT (anti-freeze): this callback does NOT call the LLM. During training it only
accumulates metrics (pure numpy, microseconds) and writes the relevant snapshots to a
JSONL. The Obsidian notes are generated AFTER training, by `writer.process_run`, so the
PPO loop never freezes waiting on Ollama.

Two-layer frequency control:
1. Cadence: only consider collecting every `write_every_steps`.
2. Novelty filter: within the cadence, only write if some key metric varied beyond
   `novelty_threshold` (relative) vs. the last written snapshot.
"""
from typing import Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from instrumentation.stats_tracker import StatsTracker
from writer.snapshot_log import SnapshotLog


# Métricas que o filtro de novidade observa para decidir se "mudou o suficiente".
_NOVELTY_KEYS = [
    "mean_reward",
    "kills_per_episode",
    "shooting_accuracy",
    "damage_taken",
    "mean_episode_length",
    "action_entropy_normalized",
    "distance_per_episode",
    "success_rate",
]


def _is_novel(current: Dict, previous: Optional[Dict], threshold: float) -> bool:
    if previous is None:
        return True
    for k in _NOVELTY_KEYS:
        cur = float(current.get(k, 0.0))
        prev = float(previous.get(k, 0.0))
        denom = abs(prev) if abs(prev) > 1e-6 else 1.0
        if abs(cur - prev) / denom >= threshold:
            return True
    return False


class DoomDocumentationCallback(BaseCallback):
    def __init__(
        self,
        tracker: StatsTracker,
        log: SnapshotLog,
        write_every_steps: int,
        novelty_threshold: float,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.tracker = tracker
        self.log = log
        self.write_every_steps = write_every_steps
        self.novelty_threshold = novelty_threshold
        self._last_write_step = 0
        self._last_written_snapshot: Optional[Dict] = None
        # Dedup of the map geometry: walls (~tens of KB) are static per map, so we
        # only keep them on the first logged snapshot of each map (not every one).
        self._last_walls_token = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", np.array([]))
        self.tracker.update(infos, actions)

        if self.num_timesteps - self._last_write_step < self.write_every_steps:
            return True

        snapshot = self.tracker.snapshot(self.num_timesteps)

        # Need at least a few episodes for the note to make sense.
        if snapshot["episodes"] == 0:
            return True

        if _is_novel(snapshot, self._last_written_snapshot, self.novelty_threshold):
            # Drop repeated map geometry: keep walls only when they change (per map).
            walls = snapshot.get("map_walls") or []
            token = (len(walls), tuple(walls[0]) if walls else None)
            if walls and token == self._last_walls_token:
                snapshot = {**snapshot, "map_walls": []}
            elif walls:
                self._last_walls_token = token
            self.log.append(snapshot)  # local write, no LLM -> doesn't block
            if self.verbose:
                print(
                    f"[doc] step={self.num_timesteps} snapshot #{self.log.count} "
                    f"collected (reward={snapshot['mean_reward']:.2f}, "
                    f"kills/ep={snapshot['kills_per_episode']:.2f}, "
                    f"accuracy={snapshot['shooting_accuracy']:.0%})"
                )
            self._last_written_snapshot = snapshot
        else:
            if self.verbose:
                print(f"[doc] step={self.num_timesteps} no novelty — skipping.")

        self._last_write_step = self.num_timesteps
        self.tracker.reset_window()
        return True


class BootTimingCallback(BaseCallback):
    """Wall-clock probe for the learn() boot path.

    Prints the time from training_start to the FIRST env step. Motivation: a ~70-minute
    pre-stepping stall was observed once (SB3's cumulative fps read 16 while the marginal
    rate was ~440 steps/s) and was impossible to localise after the fact — this pins
    whether a slow chunk lost its time before stepping (env boot, normalize restore,
    MPS warmup) or during the rollout itself.
    """

    def __init__(self):
        super().__init__()
        self._t_start: Optional[float] = None
        self._first_step_logged = False

    def _on_training_start(self) -> None:
        import time
        self._t_start = time.monotonic()
        print(f"[boot] training_start (steps so far: {self.model.num_timesteps:,})")

    def _on_step(self) -> bool:
        if not self._first_step_logged and self._t_start is not None:
            import time
            dt = time.monotonic() - self._t_start
            print(f"[boot] first env step after {dt:.1f}s")
            self._first_step_logged = True
        return True
