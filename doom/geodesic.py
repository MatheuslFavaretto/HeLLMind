"""Geodesic distance-to-exit field from WAD geometry (BFS over a wall-aware grid).

Why: the exit-proximity shaping used EUCLIDEAN distance. In a maze that creates local
minima — whenever the real route requires walking AWAY from the exit in straight-line
terms (around a wall), the gradient pins the agent against that wall. Measured symptom:
exit_progress stuck at ~9% after ~3M MAP01 steps across three exit hunts.

This module rasterises the map's WALLS (one-sided linedefs + blocking two-sided ones)
onto a coarse grid and BFS-floods distance from the exit cell. The result approximates
distance ALONG WALKABLE PATHS, so "closer" always means "closer along a real route".

    from doom.geodesic import distance_field, geodesic_distance
    field = distance_field(wad_path, "MAP01", exit_xy)     # {(cx,cy): map-units} cached
    d = geodesic_distance(field, x, y, cell=GRID_CELL)     # euclidean-fallback lookup

Pure stdlib; the field is computed once per (wad, map) per process (lru_cache).
"""
from __future__ import annotations

import struct
from collections import deque
from functools import lru_cache
from typing import Dict, Optional, Tuple

from doom.wad_doors import _lump_dir

GRID_CELL = 64  # map units per grid cell (Doom corridors are ≥64 wide by convention)

_Walls = Tuple[Tuple[int, int, int, int], ...]   # (x1, y1, x2, y2) per wall segment


def map_walls(wad_path: str, map_name: str) -> _Walls:
    """Wall segments of a map: one-sided linedefs (no back sidedef) + ML_BLOCKING lines.
    Doors are NOT walls (auto-USE opens them on contact), so door specials are skipped."""
    from doom.wad_doors import DOOR_SPECIALS
    try:
        with open(wad_path, "rb") as f:
            data = f.read()
    except OSError:
        return ()
    lumps = _lump_dir(data)
    try:
        start = next(i for i, (n, _, _) in enumerate(lumps) if n == map_name)
    except StopIteration:
        return ()
    window = lumps[start: start + 12]
    ld = next(((o, s) for n, o, s in window if n == "LINEDEFS"), None)
    vx = next(((o, s) for n, o, s in window if n == "VERTEXES"), None)
    if not ld or not vx:
        return ()
    verts = [struct.unpack("<hh", data[vx[0] + j * 4: vx[0] + j * 4 + 4])
             for j in range(vx[1] // 4)]
    walls = []
    for j in range(ld[1] // 14):
        rec = data[ld[0] + j * 14: ld[0] + j * 14 + 14]
        if len(rec) < 14:
            break
        v1, v2, flags, special, _tag, _front, back = struct.unpack("<7H", rec)
        if v1 >= len(verts) or v2 >= len(verts):
            continue
        one_sided = back == 0xFFFF
        blocking = bool(flags & 0x0001)  # ML_BLOCKING
        if (one_sided or blocking) and special not in DOOR_SPECIALS:
            (x1, y1), (x2, y2) = verts[v1], verts[v2]
            walls.append((x1, y1, x2, y2))
    return tuple(walls)


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    """True if segment AB strictly crosses segment CD (orientation test)."""
    def orient(px, py, qx, qy, rx, ry):
        v = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        return 0 if v == 0 else (1 if v > 0 else -1)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return o1 != o2 and o3 != o4


@lru_cache(maxsize=8)
def distance_field(wad_path: str, map_name: str,
                   exit_xy: Tuple[int, int],
                   cell: int = GRID_CELL) -> Dict[Tuple[int, int], float]:
    """BFS distance (in map units) from the exit over the walkable grid.

    A move between adjacent cell centers is blocked when it crosses any wall segment.
    Cells unreachable from the exit are absent (callers fall back to euclidean).
    Cached per (wad, map, exit) — costs a few seconds once per process."""
    walls = map_walls(wad_path, map_name)
    if not walls:
        return {}
    xs = [w[0] for w in walls] + [w[2] for w in walls]
    ys = [w[1] for w in walls] + [w[3] for w in walls]
    cx0, cx1 = min(xs) // cell, max(xs) // cell
    cy0, cy1 = min(ys) // cell, max(ys) // cell

    # Bucket walls by the grid cells their bounding box touches → ~O(1) walls per move.
    buckets: Dict[Tuple[int, int], list] = {}
    for w in walls:
        x1, y1, x2, y2 = w
        for gx in range(min(x1, x2) // cell - 1, max(x1, x2) // cell + 2):
            for gy in range(min(y1, y2) // cell - 1, max(y1, y2) // cell + 2):
                buckets.setdefault((gx, gy), []).append(w)

    def passable(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Can the agent move between the centers of adjacent cells a→b?"""
        ax, ay = a[0] * cell + cell // 2, a[1] * cell + cell // 2
        bx, by = b[0] * cell + cell // 2, b[1] * cell + cell // 2
        for w in buckets.get(a, []) + buckets.get(b, []):
            if _segments_cross(ax, ay, bx, by, w[0], w[1], w[2], w[3]):
                return False
        return True

    # Seed: the exit linedef midpoint sits ON a wall (exit switches are walls), so seed
    # every passable cell in a small neighbourhood of it.
    ex, ey = exit_xy
    ecx, ecy = ex // cell, ey // cell
    dist: Dict[Tuple[int, int], float] = {}
    q: deque = deque()
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            c = (ecx + ox, ecy + oy)
            if cx0 - 1 <= c[0] <= cx1 + 1 and cy0 - 1 <= c[1] <= cy1 + 1:
                dist[c] = 0.0
                q.append(c)
    while q:
        c = q.popleft()
        for nb in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
            if nb in dist or not (cx0 - 1 <= nb[0] <= cx1 + 1 and cy0 - 1 <= nb[1] <= cy1 + 1):
                continue
            if passable(c, nb):
                dist[nb] = dist[c] + cell
                q.append(nb)
    return dist


def geodesic_distance(field: Dict[Tuple[int, int], float], x: float, y: float,
                      cell: int = GRID_CELL) -> Optional[float]:
    """Distance-to-exit (map units) for a world position, or None when the position's
    cell isn't in the field (outside / unreachable) — callers fall back to euclidean."""
    return field.get((int(x) // cell, int(y) // cell))
