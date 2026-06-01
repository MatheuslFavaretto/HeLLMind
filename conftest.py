"""Configuração compartilhada dos testes.

A presença deste arquivo na raiz faz o pytest colocar o diretório do projeto no
sys.path, então `from writer... import ...` funciona sem instalar o pacote.

Fornece também uma fábrica de `info["doom"]` sintético, para testar o StatsTracker
e afins SEM precisar subir o ViZDoom (rápido e determinístico em CI).
"""
from typing import Optional, Tuple

import pytest

from instrumentation.game_vars import LEVELS, MONOTONIC


def _make_doom_info(
    action: int,
    *,
    hits: float = 0.0,
    kills: float = 0.0,
    damage_dealt: float = 0.0,
    damage_taken: float = 0.0,
    distance: float = 0.0,
    pos: Tuple[float, float] = (0.0, 0.0),
    weapon: int = 2,
    health: float = 100.0,
    ammo: float = 50.0,
    success: Optional[bool] = None,
    map_name: Optional[str] = None,
    episode: Optional[dict] = None,
) -> dict:
    deltas = {k: 0.0 for k in MONOTONIC}
    deltas["hitcount"] = hits
    deltas["killcount"] = kills
    deltas["damagecount"] = damage_dealt
    deltas["damage_taken"] = damage_taken
    deltas["distance"] = distance
    levels = {k: 0.0 for k in LEVELS}
    levels["health"] = health
    levels["ammo2"] = ammo
    levels["position_x"] = pos[0]
    levels["position_y"] = pos[1]
    levels["selected_weapon"] = weapon
    doom = {"deltas": deltas, "levels": levels, "action": action}
    if success is not None:
        doom["success"] = success
    info = {"doom": doom}
    if map_name:
        info["map"] = map_name
    if episode is not None:
        info["episode"] = episode
    return info


@pytest.fixture
def make_doom_info():
    """Fábrica de info sintético do env (parametrizável por teste)."""
    return _make_doom_info
