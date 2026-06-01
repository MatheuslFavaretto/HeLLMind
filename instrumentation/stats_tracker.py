"""Acumula sinal do ViZDoom ao longo de uma janela e produz um snapshot resumido.

Filosofia: reportar DELTAS (o que mudou desde a última nota), não totais, pra que
as notas não virem 'os números cresceram de novo'.
"""
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from instrumentation.action_stats import (
    action_distribution,
    action_entropy,
    max_entropy,
)
from instrumentation.game_vars import LEVELS

# Tamanho da célula (em unidades do mapa) ao discretizar a posição para estimar
# COBERTURA do mapa (quantas células distintas o agente pisou).
COVERAGE_CELL = 96.0


class StatsTracker:
    def __init__(self, button_names: List[str]) -> None:
        self.button_names = button_names
        self.n_actions = len(button_names)
        # Geometria do mapa (paredes) — estática; persiste entre janelas.
        self.map_walls: list = []
        self.reset_window()

    def reset_window(self) -> None:
        # Somatório dinâmico: aceita qualquer chave de delta vinda do env
        # (killcount, hitcount, ..., e também "distance").
        self.delta_sums: Dict[str, float] = {}
        self.level_samples: Dict[str, List[float]] = {n: [] for n in LEVELS}
        self.action_counts = np.zeros(self.n_actions, dtype=np.float64)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.steps_in_window = 0
        # Cobertura/caminho: quantas vezes o agente pisou em cada célula (heatmap).
        self.cell_counts: Counter = Counter()
        self.attack_actions = 0  # nº de ações de ataque (p/ precisão de tiro)
        # Campanha: mapa atual e contagem de episódios "completados".
        self.current_map: str = ""
        self.episodes_done = 0
        self.episodes_success = 0

    # ------------------------------------------------------------------
    def update(self, infos: List[dict], actions: np.ndarray) -> None:
        """Chamado a cada passo do vec env, para todos os envs em paralelo."""
        for info, act in zip(infos, actions):
            doom = info.get("doom")
            if doom is not None:
                for k, v in doom["deltas"].items():
                    self.delta_sums[k] = self.delta_sums.get(k, 0.0) + float(v)
                for n in LEVELS:
                    self.level_samples[n].append(doom["levels"][n])
                self.action_counts[int(doom["action"])] += 1
                # Cobertura/caminho: discretiza a posição numa grade e conta visitas.
                px = doom["levels"].get("position_x", 0.0)
                py = doom["levels"].get("position_y", 0.0)
                self.cell_counts[
                    (round(px / COVERAGE_CELL), round(py / COVERAGE_CELL))
                ] += 1
                walls = doom.get("walls")
                if walls:  # enviado 1x por mapa; guardamos p/ o minimapa
                    self.map_walls = walls
            if info.get("map"):
                self.current_map = info["map"]
            self.steps_in_window += 1

            ep = info.get("episode")
            if ep is not None:  # Monitor finalizou um episódio
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(int(ep["l"]))
                self.episodes_done += 1
                if doom is not None and doom.get("success"):
                    self.episodes_success += 1

    # ------------------------------------------------------------------
    def snapshot(self, num_timesteps: int) -> Dict[str, Any]:
        """Resumo da janela atual. Tudo aqui vira contexto para o LLM."""
        n_eps = len(self.episode_rewards)
        d = self.delta_sums
        shots = float(d.get("hitcount", 0.0))  # tiros que acertaram
        # 'attack' costuma ser o botão de tiro; calculamos acertos por ataque.
        attack_idx = next(
            (i for i, n in enumerate(self.button_names) if "ATTACK" in n.upper()),
            None,
        )
        attack_count = (
            float(self.action_counts[attack_idx]) if attack_idx is not None else 0.0
        )
        accuracy = (shots / attack_count) if attack_count > 0 else 0.0
        misses = max(0.0, attack_count - shots)

        def _mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        def _min(xs):
            return float(np.min(xs)) if xs else 0.0

        snap = {
            "num_timesteps": int(num_timesteps),
            "steps_in_window": int(self.steps_in_window),
            "episodes": n_eps,
            "map": self.current_map,
            "success_rate": (
                self.episodes_success / self.episodes_done
                if self.episodes_done
                else 0.0
            ),
            "mean_reward": _mean(self.episode_rewards),
            "mean_episode_length": _mean(self.episode_lengths),
            "min_episode_length": _min(self.episode_lengths),
            "max_episode_length": float(max(self.episode_lengths)) if self.episode_lengths else 0.0,
            # contadores (deltas na janela)
            "kills": d.get("killcount", 0.0),
            "hits_landed": d.get("hitcount", 0.0),
            "hits_taken": d.get("hits_taken", 0.0),
            "damage_dealt": d.get("damagecount", 0.0),
            "damage_taken": d.get("damage_taken", 0.0),
            "deaths": d.get("deathcount", 0.0),
            "items_collected": d.get("itemcount", 0.0),
            # pontaria (acertos x erros)
            "shots_fired": float(attack_count),
            "shots_hit": shots,
            "shots_missed": misses,
            "shooting_accuracy": float(accuracy),
            "kills_per_episode": (d.get("killcount", 0.0) / n_eps) if n_eps else 0.0,
            # exploração / caminho percorrido
            "distance_traveled": d.get("distance", 0.0),
            "distance_per_episode": (d.get("distance", 0.0) / n_eps) if n_eps else 0.0,
            "cells_visited": len(self.cell_counts),
            "map_coverage": self._coverage(),
            "weapons_used": self._weapon_distribution(),
            # células [gx, gy, visitas] para renderizar o minimapa do caminho
            "path_cells": [[gx, gy, c] for (gx, gy), c in self.cell_counts.items()],
            # paredes do mapa real [x1,y1,x2,y2] (fundo do minimapa)
            "map_walls": self.map_walls,
            # níveis
            "mean_health": _mean(self.level_samples["health"]),
            "min_health": _min(self.level_samples["health"]),
            "mean_ammo": _mean(self.level_samples["ammo2"]),
            # ações
            "action_distribution": action_distribution(
                self.action_counts, self.button_names
            ),
            "action_entropy": action_entropy(self.action_counts),
            "action_entropy_normalized": (
                action_entropy(self.action_counts) / max_entropy(self.n_actions)
                if max_entropy(self.n_actions) > 0
                else 0.0
            ),
        }
        return snap

    # ------------------------------------------------------------------
    def _coverage(self) -> Dict[str, float]:
        """Estima a fração explorada: células visitadas / área do bounding box.

        Não sabemos o tamanho real do mapa, então normalizamos pela caixa
        delimitadora (bounding box) das posições vistas — uma aproximação honesta
        de 'quanto da área percorrida o agente realmente cobriu'.
        """
        xs = self.level_samples.get("position_x", [])
        ys = self.level_samples.get("position_y", [])
        n_cells = len(self.cell_counts)
        if len(xs) < 2:
            return {"cells_visited": float(n_cells), "explored_fraction": 0.0}
        span_x = (max(xs) - min(xs)) / COVERAGE_CELL
        span_y = (max(ys) - min(ys)) / COVERAGE_CELL
        bbox_cells = max(1.0, (span_x + 1.0) * (span_y + 1.0))
        return {
            "cells_visited": float(n_cells),
            "bbox_cells": float(round(bbox_cells, 1)),
            "explored_fraction": float(min(1.0, n_cells / bbox_cells)),
        }

    def _weapon_distribution(self) -> Dict[str, float]:
        """Fração do tempo com cada arma selecionada (slot do ViZDoom)."""
        samples = self.level_samples.get("selected_weapon", [])
        if not samples:
            return {}
        counts = Counter(int(s) for s in samples)
        total = float(sum(counts.values()))
        return {f"slot_{k}": v / total for k, v in sorted(counts.items())}
