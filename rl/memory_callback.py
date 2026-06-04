"""Records episode-end events into the persistent memory (Phase 1).

A separate, tiny callback so memory works independently of the documentation path.
On each episode end it appends one structured event (death/success/timeout + context)
to the MemoryStore. This is the only training-time touchpoint and it's cheap (one
appended line per finished episode), so it stays within the ±2% FPS budget.
"""
from stable_baselines3.common.callbacks import BaseCallback

from writer.memory_store import MemoryStore


class MemoryRecorderCallback(BaseCallback):
    def __init__(self, memory: MemoryStore, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.memory = memory

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:  # only act when an episode finished
                continue
            doom = info.get("doom") or {}
            levels = doom.get("levels", {}) or {}
            health = float(levels.get("health", 0.0))
            # The env tags the terminal type reliably (via is_player_dead()).
            terminal = doom.get("terminal")
            if terminal:
                etype = terminal
            elif doom.get("success"):
                etype = "success"
            else:
                etype = "death" if health <= 0 else "timeout"
            try:
                px, py = doom.get("final_pos", [0, 0])
                region = f"{round(px / 512)}x{round(py / 512)}"
                self.memory.record_event({
                    "type": etype,
                    "timesteps": int(self.num_timesteps),
                    "map": info.get("map", ""),
                    "reward": round(float(ep["r"]), 2),
                    "length": int(ep["l"]),
                    "health": round(health),
                    "ammo": round(float(levels.get("ammo2", 0.0))),
                    "coverage": int(doom.get("coverage_cells", 0)),
                    "weapon": int(levels.get("selected_weapon", 0.0)),
                    "region": region,
                    "nearest_enemy": doom.get("nearest_enemy", ""),
                })
            except Exception:
                pass  # memory never crashes training
        return True
