"""Reads the ViZDoom map geometry (wall segments) for the real minimap.

With `set_sectors_info_enabled(True)`, each state carries `state.sectors`, and each
sector has `lines` (segments with x1,y1,x2,y2 and is_blocking). Collecting the blocking
lines gives the real level outline — which we draw as the minimap background, with the
agent's path on top (instead of "loose squares").
"""
from typing import List


def read_wall_segments(game, blocking_only: bool = True, max_segments: int = 5000) -> List[List[float]]:
    """Return [[x1,y1,x2,y2], ...] of the current map's walls (world units, same as
    POSITION_X/Y). Empty if sector info isn't available."""
    state = game.get_state()
    sectors = getattr(state, "sectors", None) if state is not None else None
    if not sectors:
        return []
    segs: List[List[float]] = []
    for sec in sectors:
        for ln in sec.lines:
            if blocking_only and not ln.is_blocking:
                continue
            segs.append([float(ln.x1), float(ln.y1), float(ln.x2), float(ln.y2)])
            if len(segs) >= max_segments:
                return segs
    return segs
