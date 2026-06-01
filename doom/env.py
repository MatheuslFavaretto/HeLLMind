"""Wrapper Gymnasium em torno do ViZDoom.

Cada passo emite, além de obs/reward, um dicionário `info["doom"]` com os deltas
dos contadores e os níveis instantâneos (vida/munição). É esse sinal que alimenta
o StatsTracker e, por fim, as notas do Obsidian.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from doom.geometry import read_wall_segments
from instrumentation.game_vars import LEVELS, MONOTONIC, TRACKED_VARS, VAR_NAMES

# --- Reward shaping (acertos/erros e perda de desempenho) ---
# Recompensa por tiro que ACERTOU (delta de HITCOUNT na janela do frame_skip).
HIT_REWARD = 1.0
# Punição por ATACAR e NÃO acertar nada. Menor que HIT_REWARD de propósito:
# punição alta demais ensina o agente a parar de atirar (vira passivo).
MISS_PENALTY = 0.25
# Punição por PERDER DESEMPENHO: tomar dano (por ponto) e morrer.
DAMAGE_TAKEN_PENALTY = 0.05
DEATH_PENALTY = 5.0


class DoomEnv(gym.Env):
    """Ambiente single-process. Use a factory `make_doom_env` com SubprocVecEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "defend_the_center",
        frame_skip: int = 4,
        resolution: Tuple[int, int] = (84, 84),
        window_visible: bool = False,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.width, self.height = resolution

        game = vzd.DoomGame()
        cfg = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        game.load_config(cfg)
        game.set_window_visible(window_visible)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        # Janela visível pede uma resolução maior para dar pra enxergar.
        game.set_screen_resolution(
            vzd.ScreenResolution.RES_640X480
            if window_visible
            else vzd.ScreenResolution.RES_160X120
        )
        # Sobrescrevemos as variáveis do cfg pelo nosso conjunto rico.
        game.set_available_game_variables(TRACKED_VARS)
        game.set_sectors_info_enabled(True)  # geometria do mapa p/ o minimapa real
        game.init()
        self.game = game
        self._walls_pending = True  # envia as paredes uma vez (mapa fixo no cenário)

        self.buttons: List[vzd.Button] = game.get_available_buttons()
        self.button_names: List[str] = [b.name for b in self.buttons]
        n = len(self.buttons)
        # Ações discretas one-hot: uma ação por botão disponível.
        self.actions: List[List[int]] = [
            [1 if i == j else 0 for i in range(n)] for j in range(n)
        ]

        # Índice do botão de ATAQUE (p/ saber quando o agente "errou" um tiro).
        self._attack_idx = next(
            (i for i, nm in enumerate(self.button_names) if "ATTACK" in nm.upper()),
            None,
        )

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )
        self._last_vars: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    def _read_raw_vars(self) -> Dict[str, float]:
        state = self.game.get_state()
        if state is None:
            return self._last_vars or {n: 0.0 for n in VAR_NAMES}
        vals = state.game_variables
        return {VAR_NAMES[i]: float(vals[i]) for i in range(len(VAR_NAMES))}

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        frame = state.screen_buffer  # (120, 160) uint8
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]

    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self._last_vars = self._read_raw_vars()
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        buttons = self.actions[int(action)]
        base_reward = self.game.make_action(buttons, self.frame_skip)
        done = self.game.is_episode_finished()

        if not done:
            raw = self._read_raw_vars()
            deltas = {n: max(0.0, raw[n] - self._last_vars[n]) for n in MONOTONIC}
            # Distância percorrida neste passo (p/ medir exploração do mapa).
            dx = raw["position_x"] - self._last_vars["position_x"]
            dy = raw["position_y"] - self._last_vars["position_y"]
            deltas["distance"] = float((dx * dx + dy * dy) ** 0.5)
            levels = {n: raw[n] for n in LEVELS}
            self._last_vars = raw
            obs = self._get_obs()
        else:
            # Estado final não tem screen/vars; usamos o último conhecido.
            deltas = {n: 0.0 for n in MONOTONIC}
            deltas["distance"] = 0.0
            levels = {n: self._last_vars[n] for n in LEVELS}
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Shaping: + por acerto; - por errar; - por perder desempenho (dano/morte).
        reward = base_reward + HIT_REWARD * deltas["hitcount"]
        attacked = self._attack_idx is not None and int(action) == self._attack_idx
        if attacked and deltas["hitcount"] == 0 and not done:
            reward -= MISS_PENALTY
        reward -= DAMAGE_TAKEN_PENALTY * deltas["damage_taken"]
        if done and levels["health"] <= 0:
            reward -= DEATH_PENALTY

        doom = {"deltas": deltas, "levels": levels, "action": int(action)}
        # Geometria do mapa: enviada UMA vez (não a cada passo — não pesa no loop).
        if self._walls_pending:
            doom["walls"] = read_wall_segments(self.game)
            self._walls_pending = False
        return obs, float(reward), done, False, {"doom": doom}

    def close(self) -> None:
        self.game.close()


def make_doom_env(
    scenario: str,
    frame_skip: int,
    resolution: Tuple[int, int],
    seed: int,
    rank: int,
    window_visible: bool = False,
):
    """Factory para SubprocVecEnv. Cada subprocesso recebe um seed distinto.

    Não envolvemos com Monitor aqui: o VecMonitor (aplicado uma vez sobre o
    vec env) já injeta info["episode"], evitando o aviso de Monitor duplicado.
    """

    def _init():
        env = DoomEnv(
            scenario=scenario,
            frame_skip=frame_skip,
            resolution=resolution,
            window_visible=window_visible,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def probe_env_metadata(
    scenario: str, frame_skip: int, resolution: Tuple[int, int]
) -> Dict[str, Any]:
    """Cria um env temporário só para descobrir nomes de botões e nº de ações.

    Útil para o StatsTracker (rótulos da distribuição de ações) sem precisar
    iniciar o treino inteiro.
    """
    env = DoomEnv(scenario=scenario, frame_skip=frame_skip, resolution=resolution)
    meta = {
        "button_names": list(env.button_names),
        "num_actions": int(env.action_space.n),
    }
    env.close()
    return meta
