"""Modo CAMPANHA: jogar mapas completos de um WAD (Doom 1 / Freedoom), em ordem.

Diferente dos cenários do ViZDoom (objetivo único e curto), aqui carregamos um
WAD com mapas de verdade (MAP01.., ou E1M1.. no doom.wad original) e treinamos o
agente a "completar e seguir para o próximo".

Decisões desta POC (escolhidas pelo usuário):
- "Completar" = sobreviver E/OU matar X inimigos (critério de sucesso, usado em
  reward shaping e logging).
- Avanço entre mapas = SEQUENCIAL POR TIMESTEPS (ver MapCurriculumCallback).

O env emite `info["doom"]` no MESMO formato do env de cenário (deltas/levels/action)
para reaproveitar o StatsTracker, e adiciona `info["map"]` e `info["doom"]["success"]`.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from doom.env import HIT_REWARD, MISS_PENALTY
from doom.geometry import read_wall_segments
from instrumentation.game_vars import LEVELS, MONOTONIC, TRACKED_VARS, VAR_NAMES

# Botões necessários para ATRAVESSAR um mapa (mover, virar, atirar, usar portas).
CAMPAIGN_BUTTONS = [
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.USE,                  # abrir portas / acionar a saída do nível
    vzd.Button.SPEED,
    vzd.Button.SELECT_NEXT_WEAPON,   # trocar de arma (gera variedade de armas)
]


def default_wad() -> str:
    """WAD padrão: o freedoom2.wad que vem embutido no ViZDoom (gratuito/legal).

    Ele fica na raiz do pacote vizdoom (ao lado de scenarios/), não dentro de
    scenarios/.
    """
    return os.path.join(os.path.dirname(vzd.scenarios_path), "freedoom2.wad")


class CampaignDoomEnv(gym.Env):
    """Joga um mapa completo de um WAD. O mapa pode ser trocado em runtime."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        wad_path: str,
        doom_map: str = "MAP01",
        frame_skip: int = 4,
        resolution: Tuple[int, int] = (84, 84),
        episode_timeout: int = 2100,   # ticks (~60s a 35fps) para não travar
        kills_to_clear: int = 5,
        window_visible: bool = False,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.width, self.height = resolution
        self.kills_to_clear = kills_to_clear
        self._current_map = doom_map
        self._pending_map: Optional[str] = None

        game = vzd.DoomGame()
        # IWAD completo (freedoom2.wad / doom.wad) -> game path; mapas via set_doom_map.
        game.set_doom_game_path(wad_path)
        game.set_doom_map(doom_map)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(
            vzd.ScreenResolution.RES_640X480
            if window_visible
            else vzd.ScreenResolution.RES_160X120
        )
        game.set_window_visible(window_visible)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_episode_timeout(episode_timeout)
        game.set_available_buttons(CAMPAIGN_BUTTONS)
        game.set_available_game_variables(TRACKED_VARS)
        game.set_sectors_info_enabled(True)  # geometria do mapa p/ o minimapa real
        # Recompensas internas do ViZDoom: vivo é bom, morrer é ruim.
        game.set_living_reward(0.01)
        game.set_death_penalty(100.0)
        game.init()
        self.game = game

        self.button_names: List[str] = [b.name for b in CAMPAIGN_BUTTONS]
        n = len(CAMPAIGN_BUTTONS)
        self.actions: List[List[int]] = [
            [1 if i == j else 0 for i in range(n)] for j in range(n)
        ]
        self._attack_idx = next(
            (i for i, nm in enumerate(self.button_names) if "ATTACK" in nm.upper()),
            None,
        )

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )
        self._last_vars: Optional[Dict[str, float]] = None
        self._walls_pending = True  # envia paredes 1x por mapa (re-arma na troca)

    # ------------------------------------------------------------------
    def set_map(self, doom_map: str) -> None:
        """Agenda troca de mapa; aplica no próximo reset (usado pelo currículo)."""
        self._pending_map = doom_map

    @property
    def current_map(self) -> str:
        return self._current_map

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
        frame = cv2.resize(
            state.screen_buffer, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]

    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        if self._pending_map is not None and self._pending_map != self._current_map:
            self.game.set_doom_map(self._pending_map)
            self._current_map = self._pending_map
            self._pending_map = None
            self._walls_pending = True  # mapa novo -> reenvia a geometria
        self.game.new_episode()
        self._last_vars = self._read_raw_vars()
        return self._get_obs(), {"map": self._current_map}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        buttons = self.actions[int(action)]
        base_reward = self.game.make_action(buttons, self.frame_skip)
        done = self.game.is_episode_finished()

        if not done:
            raw = self._read_raw_vars()
            deltas = {n: max(0.0, raw[n] - self._last_vars[n]) for n in MONOTONIC}
            dx = raw["position_x"] - self._last_vars["position_x"]
            dy = raw["position_y"] - self._last_vars["position_y"]
            deltas["distance"] = float((dx * dx + dy * dy) ** 0.5)
            levels = {n: raw[n] for n in LEVELS}
            self._last_vars = raw
            obs = self._get_obs()
        else:
            deltas = {n: 0.0 for n in MONOTONIC}
            deltas["distance"] = 0.0
            levels = {n: self._last_vars[n] for n in LEVELS}
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Reward shaping: kills positivos, dano tomado negativo, + pontaria.
        shaped = base_reward + 5.0 * deltas["killcount"] - 0.1 * deltas["damage_taken"]
        shaped += HIT_REWARD * deltas["hitcount"]
        attacked = self._attack_idx is not None and int(action) == self._attack_idx
        if attacked and deltas["hitcount"] == 0 and not done:
            shaped -= MISS_PENALTY

        # "Completou" = terminou o episódio sem ter morrido (chegou na saída / sobreviveu
        # ao timeout) OU já bateu a cota de kills no episódio.
        alive = levels["health"] > 0
        kills_total = self._last_vars.get("killcount", 0.0) if self._last_vars else 0.0
        success = bool(done and alive) or (kills_total >= self.kills_to_clear)
        if done and alive:
            shaped += 100.0  # bônus por concluir o mapa vivo

        doom = {
            "deltas": deltas,
            "levels": levels,
            "action": int(action),
            "success": success,
        }
        if self._walls_pending:
            doom["walls"] = read_wall_segments(self.game)
            self._walls_pending = False
        return obs, float(shaped), done, False, {"map": self._current_map, "doom": doom}

    def close(self) -> None:
        self.game.close()


def make_campaign_env(
    wad_path: str,
    doom_map: str,
    frame_skip: int,
    resolution: Tuple[int, int],
    episode_timeout: int,
    kills_to_clear: int,
    seed: int,
    rank: int,
    window_visible: bool = False,
):
    """Factory para SubprocVecEnv/DummyVecEnv."""

    def _init():
        env = CampaignDoomEnv(
            wad_path=wad_path,
            doom_map=doom_map,
            frame_skip=frame_skip,
            resolution=resolution,
            episode_timeout=episode_timeout,
            kills_to_clear=kills_to_clear,
            window_visible=window_visible,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def campaign_metadata(wad_path: str, doom_map: str) -> Dict[str, Any]:
    """Descobre nomes de botões/nº de ações sem subir o treino."""
    env = CampaignDoomEnv(wad_path=wad_path, doom_map=doom_map)
    meta = {"button_names": list(env.button_names), "num_actions": int(env.action_space.n)}
    env.close()
    return meta
