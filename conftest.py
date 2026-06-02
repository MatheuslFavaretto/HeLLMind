"""Shared test configuration.

Having this file at the root makes pytest put the project directory on sys.path, so
`from writer... import ...` works without installing the package.

It also provides a synthetic `info["doom"]` factory, to test the StatsTracker and
friends WITHOUT booting ViZDoom (fast and deterministic in CI).
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
    """Synthetic env-info factory (parameterizable per test)."""
    return _make_doom_info
