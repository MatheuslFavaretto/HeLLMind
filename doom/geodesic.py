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

# (x1, y1, x2, y2, front_floor, back_floor) per segment; floors None = hard wall,
# numeric = DIRECTIONAL step (drop always passable, climb only ≤ MAX_STEP).
_Walls = Tuple[Tuple[int, int, int, int, Optional[int], Optional[int]], ...]


# Lift/platform specials: the floor MOVES, so a step across these lines is traversable
# even when the static heights differ (W1/WR/S1/SR lower-lift + raise variants).
LIFT_SPECIALS = frozenset({10, 21, 62, 88, 120, 121, 122, 123})

MAX_STEP = 24      # Doom's maximum climbable floor step, in map units
MIN_HEADROOM = 56  # player height + margin: can't pass where ceiling-floor is lower

# Exit specials must stay PASSABLE: the exit switch often sits on its own step
# (freedoom2 MAP01: a 48u step ON the exit line) and USE activates it from below.
from doom.wad_doors import EXIT_SPECIALS as _EXIT_SPECIALS


def map_walls(wad_path: str, map_name: str) -> _Walls:
    """Wall segments of a map, HEIGHT-AWARE:
    - one-sided linedefs + ML_BLOCKING lines (classic walls)
    - two-sided lines whose floor step exceeds MAX_STEP (cliffs — found the hard way:
      the 2D field guided the agent into a 120u cliff at 90% of MAP01's route, where it
      paced until timeout for four straight evals)
    - two-sided lines without MIN_HEADROOM of clearance (crushers/slits)
    Doors are NOT walls (auto-USE opens them); lift lines are NOT walls (the floor
    moves); exit-special lines are NOT walls (USE reaches the switch)."""
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
    sd = next(((o, s) for n, o, s in window if n == "SIDEDEFS"), None)
    sec = next(((o, s) for n, o, s in window if n == "SECTORS"), None)
    if not ld or not vx:
        return ()
    verts = [struct.unpack("<hh", data[vx[0] + j * 4: vx[0] + j * 4 + 4])
             for j in range(vx[1] // 4)]
    # SIDEDEFS (30 bytes) → owning sector; SECTORS (26 bytes) → floor/ceiling heights.
    side_sector = []
    if sd:
        side_sector = [struct.unpack("<h", data[sd[0] + j * 30 + 28: sd[0] + j * 30 + 30])[0]
                       for j in range(sd[1] // 30)]
    sector_h = []
    if sec:
        sector_h = [struct.unpack("<hh", data[sec[0] + j * 26: sec[0] + j * 26 + 4])
                    for j in range(sec[1] // 26)]

    skip_specials = DOOR_SPECIALS | LIFT_SPECIALS | _EXIT_SPECIALS
    walls = []
    for j in range(ld[1] // 14):
        rec = data[ld[0] + j * 14: ld[0] + j * 14 + 14]
        if len(rec) < 14:
            break
        v1, v2, flags, special, _tag, front, back = struct.unpack("<7H", rec)
        if v1 >= len(verts) or v2 >= len(verts):
            continue
        one_sided = back == 0xFFFF
        blocking = bool(flags & 0x0001)  # ML_BLOCKING
        impassable = one_sided or blocking
        f_floor = b_floor = None
        # Height check for two-sided lines (needs SIDEDEFS+SECTORS parsed).
        if (not impassable and side_sector and sector_h
                and front < len(side_sector) and back < len(side_sector)):
            fs, bs = side_sector[front], side_sector[back]
            if 0 <= fs < len(sector_h) and 0 <= bs < len(sector_h):
                f_floor, f_ceil = sector_h[fs]
                b_floor, b_ceil = sector_h[bs]
                headroom = min(f_ceil, b_ceil) - max(f_floor, b_floor)
                if headroom < MIN_HEADROOM:
                    impassable = True
        if special in skip_specials:
            continue
        (x1, y1), (x2, y2) = verts[v1], verts[v2]
        if impassable:
            walls.append((x1, y1, x2, y2, None, None))
        elif f_floor is not None and abs(f_floor - b_floor) > MAX_STEP:
            # DIRECTIONAL step: falling is always allowed, climbing only ≤ MAX_STEP.
            # Front side is on the RIGHT of v1→v2; floors attached for the passability
            # check at BFS time. (An undirected cliff-wall model made the exit alcove a
            # 20-cell island and 'unreachable' — the player legitimately DROPS into
            # lower areas all over the route.)
            walls.append((x1, y1, x2, y2, f_floor, b_floor))
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
        x1, y1, x2, y2 = w[0], w[1], w[2], w[3]
        for gx in range(min(x1, x2) // cell - 1, max(x1, x2) // cell + 2):
            for gy in range(min(y1, y2) // cell - 1, max(y1, y2) // cell + 2):
                buckets.setdefault((gx, gy), []).append(w)

    def _orient(px, py, qx, qy, rx, ry):
        v = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        return 0 if v == 0 else (1 if v > 0 else -1)

    PLAYER_DIAMETER = 32  # radius 16: a slit narrower than the body never drops the floor

    def passable(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Can the PLAYER move from cell a's center to cell b's center?

        Hard walls block both ways. Step lines are DIRECTIONAL: dropping any height is
        fine, climbing more than MAX_STEP is not. Doom's front sidedef is on the RIGHT
        of v1→v2, so the side the player starts on follows from the orientation of a's
        center relative to the line.

        SLITS: the body has a 16u radius — the engine floors you at the HIGHEST sector
        the body touches, so a sub-32u-wide pit slit between two walkable floors never
        drops you (freedoom2 MAP01's exit walkway is separated from the exit island by
        a decorative 8u × 384u-deep slit; a point model falls in, a player walks over).
        We sort the crossings along the move and allow a >MAX_STEP climb when it
        returns, within the body width, to ≤ the pre-drop floor + MAX_STEP."""
        ax, ay = a[0] * cell + cell // 2, a[1] * cell + cell // 2
        bx, by = b[0] * cell + cell // 2, b[1] * cell + cell // 2
        seg_len = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 or 1.0

        crossings = []  # (t, src_floor, dst_floor) along the move, or hard wall
        seen = set()
        for w in buckets.get(a, []) + buckets.get(b, []):
            if id(w) in seen:
                continue
            seen.add(id(w))
            x1, y1, x2, y2, f_floor, b_floor = w
            if not _segments_cross(ax, ay, bx, by, x1, y1, x2, y2):
                continue
            if f_floor is None:
                return False  # hard wall
            # Parametric t of the intersection along the move segment.
            den = (bx - ax) * (y2 - y1) - (by - ay) * (x2 - x1)
            t = (((x1 - ax) * (y2 - y1) - (y1 - ay) * (x2 - x1)) / den) if den else 0.5
            src, dst = ((b_floor, f_floor)
                        if _orient(x1, y1, x2, y2, ax, ay) > 0
                        else (f_floor, b_floor))
            crossings.append((t, src, dst))

        # Walk the crossings in order, tracking the body's effective floor.
        crossings.sort()
        floor_now = None          # None until the first step line tells us
        drop_from = None          # (t, floor before the drop) of the most recent drop
        for t, src, dst in crossings:
            if floor_now is None:
                floor_now = src
            if dst - floor_now > MAX_STEP:
                # A climb: allowed only as the far edge of a sub-body-width slit.
                if (drop_from is not None
                        and (t - drop_from[0]) * seg_len <= PLAYER_DIAMETER
                        and dst <= drop_from[1] + MAX_STEP):
                    floor_now = dst
                    drop_from = None
                    continue
                return False
            if dst < floor_now - MAX_STEP:
                drop_from = (t, floor_now)  # falling edge: remember where we'd land from
            floor_now = dst
        return True

    # Seed: the exit midpoint cell + only the neighbours a PLAYER could step to it from.
    # An unconditional 3x3 seed put dist=0 cells inside the pit NORTH of freedoom2
    # MAP01's exit island — the whole (unescapable) pit flooded with near-zero values
    # and the agent learned to DIVE IN ('route_progress 93%' at the bottom of a -480
    # hole). Connectivity-checked seeding keeps the field on the walkable side.
    ex, ey = exit_xy
    ecx, ecy = ex // cell, ey // cell
    center = (ecx, ecy)
    dist: Dict[Tuple[int, int], float] = {center: 0.0}
    q: deque = deque([center])
    for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        c = (ecx + ox, ecy + oy)
        if (cx0 - 1 <= c[0] <= cx1 + 1 and cy0 - 1 <= c[1] <= cy1 + 1
                and passable(c, center)):   # player can walk c → exit cell
            dist[c] = 0.0
            q.append(c)
    while q:
        c = q.popleft()
        for nb in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
            if nb in dist or not (cx0 - 1 <= nb[0] <= cx1 + 1 and cy0 - 1 <= nb[1] <= cy1 + 1):
                continue
            # BFS expands exit→spawn, the REVERSE of player motion: reaching nb at
            # dist+1 means the PLAYER walks nb→c, so check passability in THAT direction
            # (directional steps: drops allowed, climbs >MAX_STEP not).
            if passable(nb, c):
                dist[nb] = dist[c] + cell
                q.append(nb)
    return dist


def geodesic_distance(field: Dict[Tuple[int, int], float], x: float, y: float,
                      cell: int = GRID_CELL) -> Optional[float]:
    """Distance-to-exit (map units) for a world position, or None when the position's
    cell isn't in the field (outside / unreachable) — callers fall back to euclidean."""
    return field.get((int(x) // cell, int(y) // cell))
