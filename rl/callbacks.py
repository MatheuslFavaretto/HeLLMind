"""Callback que coleta métricas a cada passo e SERIALIZA snapshots novos.

IMPORTANTE (anti-trava): este callback NÃO chama o LLM. Durante o treino ele só
acumula métricas (numpy puro, microssegundos) e grava os snapshots relevantes num
JSONL. As notas do Obsidian são geradas DEPOIS do treino, por `writer.process_run`,
para que o loop do PPO nunca congele esperando o Ollama.

Controle de frequência em duas camadas:
1. Cadência: só considera coletar a cada `write_every_steps`.
2. Filtro de novidade: dentro da cadência, só grava se alguma métrica-chave variou
   além de `novelty_threshold` (relativo) vs. o último snapshot gravado.
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

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", np.array([]))
        self.tracker.update(infos, actions)

        if self.num_timesteps - self._last_write_step < self.write_every_steps:
            return True

        snapshot = self.tracker.snapshot(self.num_timesteps)

        # Precisa ter visto pelo menos alguns episódios para a nota fazer sentido.
        if snapshot["episodes"] == 0:
            return True

        if _is_novel(snapshot, self._last_written_snapshot, self.novelty_threshold):
            self.log.append(snapshot)  # gravação local, sem LLM -> não trava
            if self.verbose:
                print(
                    f"[doc] step={self.num_timesteps} snapshot #{self.log.count} "
                    f"coletado (reward={snapshot['mean_reward']:.2f}, "
                    f"kills/ep={snapshot['kills_per_episode']:.2f}, "
                    f"precisão={snapshot['shooting_accuracy']:.0%})"
                )
            self._last_written_snapshot = snapshot
        else:
            if self.verbose:
                print(f"[doc] step={self.num_timesteps} sem novidade — pulando.")

        self._last_write_step = self.num_timesteps
        self.tracker.reset_window()
        return True
