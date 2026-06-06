"""Read DOOR positions straight from the WAD (the reliable way to "see" doors).

ViZDoom's runtime API only labels ACTORS (monsters/items) and exposes sector geometry
without line specials — so a door is geometrically indistinguishable from a wall/pillar at
runtime (the ceiling≈floor heuristic gives ~100 false positives on MAP01). The WAD itself,
though, marks every door with a linedef SPECIAL type. Parsing the map's LINEDEFS + VERTEXES
lumps gives the exact door positions (14 real doors on freedoom2 MAP01, not 103).

We parse the classic Doom map format (PWAD/IWAD directory → per-map lumps). No new deps.

    from doom.wad_doors import map_doors
    doors = map_doors(wad_path, "MAP01")   # -> [(x, y), ...] door midpoints, map coords
"""
import struct
from functools import lru_cache
from typing import List, Tuple

# Doom linedef SPECIAL types that are DOORS (manual/triggered/locked). Covers the common
# DR/D1/SR/S1/GR door actions across Doom + Boom; non-door specials (lifts, floors…) excluded.
DOOR_SPECIALS = {
    1, 2, 3, 4, 16, 26, 27, 28, 29, 31, 32, 33, 34, 42, 46, 50, 61, 63, 75, 76, 86, 90,
    103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 133, 134,
    135, 136, 137,
}


def _lump_dir(data: bytes):
    """Parse the WAD header + directory → list of (name, offset, size)."""
    if len(data) < 12 or data[:4] not in (b"IWAD", b"PWAD"):
        return []
    _, numlumps, diroff = struct.unpack("<4sii", data[:12])
    out = []
    for i in range(numlumps):
        rec = data[diroff + i * 16: diroff + i * 16 + 16]
        if len(rec) < 16:
            break
        off, sz, name = struct.unpack("<ii8s", rec)
        out.append((name.rstrip(b"\x00").decode("latin1"), off, sz))
    return out


@lru_cache(maxsize=16)
def map_doors(wad_path: str, map_name: str) -> Tuple[Tuple[int, int], ...]:
    """Door midpoints (x, y) in map coordinates for `map_name`. Cached per (wad, map).
    Returns an empty tuple if the WAD/map can't be parsed (so callers degrade gracefully)."""
    try:
        with open(wad_path, "rb") as f:
            data = f.read()
    except OSError:
        return ()
    lumps = _lump_dir(data)
    # Find the map marker, then its LINEDEFS/VERTEXES among the lumps that follow it.
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
    doors: List[Tuple[int, int]] = []
    for j in range(ld[1] // 14):                       # Doom-format linedef = 14 bytes
        rec = data[ld[0] + j * 14: ld[0] + j * 14 + 14]
        if len(rec) < 14:
            break
        v1, v2, _flags, special, _tag, _fr, _bk = struct.unpack("<7H", rec)
        if special in DOOR_SPECIALS and v1 < len(verts) and v2 < len(verts):
            (x1, y1), (x2, y2) = verts[v1], verts[v2]
            doors.append(((x1 + x2) // 2, (y1 + y2) // 2))
    return tuple(doors)
