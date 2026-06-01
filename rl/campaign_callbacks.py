"""Currículo de mapas: avança sequencialmente para o próximo mapa por timesteps.

Decisão do usuário: o agente treina `steps_per_map` em cada mapa e então passa ao
próximo da lista. Ao terminar a lista, repete (se loop) ou para de trocar.
A troca é aplicada via `env_method("set_map", ...)`; cada env aplica no reset.
"""
from typing import List

from stable_baselines3.common.callbacks import BaseCallback


class MapCurriculumCallback(BaseCallback):
    def __init__(
        self,
        maps: List[str],
        steps_per_map: int,
        loop_maps: bool = False,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.maps = maps
        self.steps_per_map = steps_per_map
        self.loop_maps = loop_maps
        self._map_idx = 0
        self._next_switch = steps_per_map  # primeiro switch após steps_per_map

    def _current_map(self) -> str:
        return self.maps[self._map_idx]

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_switch:
            return True

        # Hora de avançar de mapa.
        last = self._map_idx >= len(self.maps) - 1
        if last and not self.loop_maps:
            # Já estamos no último mapa e não há loop: nada a trocar.
            self._next_switch = float("inf")
            if self.verbose:
                print(f"[curriculum] último mapa ({self._current_map()}) — sem mais trocas.")
            return True

        self._map_idx = (self._map_idx + 1) % len(self.maps)
        new_map = self._current_map()
        # Agenda o novo mapa em todos os envs (aplica no próximo reset de cada um).
        self.training_env.env_method("set_map", new_map)
        self._next_switch = self.num_timesteps + self.steps_per_map
        if self.verbose:
            print(
                f"[curriculum] step={self.num_timesteps} -> trocando para {new_map} "
                f"({self._map_idx + 1}/{len(self.maps)})"
            )
        return True
