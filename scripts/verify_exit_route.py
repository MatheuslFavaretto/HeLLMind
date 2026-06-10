#!/usr/bin/env python3
"""Physically verify the geodesic route to a map's exit is traversable.

A scripted walker follows the geodesic distance field downhill (turn toward the
neighbouring cell with the lowest distance-to-exit, move forward) with auto-USE on,
exactly like the solo-with-doors training config. No learning involved — this answers
one question: *is there any structural blocker left between spawn and the exit?*

  exit reached   → the route works; training is purely a learning problem now.
                   Bonus: the success writes the TRUE exit position into exit_store.
  exit NOT reached → prints where the walker got stuck (the next blocker to fix).

    python scripts/verify_exit_route.py --map MAP01 --episodes 2
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import Config                                    # noqa: E402
from doom.campaign import CampaignDoomEnv, default_wad       # noqa: E402
from doom.geodesic import GRID_CELL, distance_field, geodesic_distance  # noqa: E402
from doom.wad_doors import map_exit                          # noqa: E402


def bearing_to(px, py, tx, ty) -> float:
    """Doom-convention bearing (degrees, 0=east, CCW) from (px,py) to (tx,ty)."""
    return math.degrees(math.atan2(ty - py, tx - px)) % 360


def pick_action(angle_diff: float) -> int:
    """Map the signed angle error to a campaign action index.
    0=FWD 1=FWD+TL 2=FWD+TR 6=TL 7=TR (fixed CAMPAIGN_ACTIONS order)."""
    if abs(angle_diff) < 15:
        return 0          # straight ahead
    if angle_diff > 0:    # target to the left
        return 1 if angle_diff < 75 else 6
    return 2 if angle_diff > -75 else 7


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map", default="MAP01")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--timeout", type=int, default=4200, help="Episode timeout (ticks).")
    args = p.parse_args()

    cfg = Config()
    wad = cfg.wad_path or default_wad()
    exit_xy = map_exit(wad, args.map)
    if not exit_xy:
        sys.exit(f"no exit linedef found for {args.map}")
    field = distance_field(wad, args.map, tuple(exit_xy))
    print(f"[route] {args.map}: exit={exit_xy}, field={len(field)} cells")

    env = CampaignDoomEnv(
        wad_path=wad, doom_map=args.map, episode_timeout=args.timeout,
        rewards={"auto_use": 1.0},          # doors open on contact — nothing else assisted
        memory_dir=cfg.memory_dir,          # a success writes the TRUE exit to exit_store
    )
    try:
        for ep in range(args.episodes):
            env.reset()
            stuck, last_pos, best = 0, None, 1e9
            for step in range(args.timeout):
                v = env._last_vars
                px, py, ang = v["position_x"], v["position_y"], v.get("angle", 0.0)
                d = geodesic_distance(field, px, py)
                if d is not None:
                    best = min(best, d)
                # Next waypoint: neighbouring cell with the lowest distance.
                cell = (int(px) // GRID_CELL, int(py) // GRID_CELL)
                nbs = [(cell[0] + dx, cell[1] + dy)
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                nxt = min((n for n in nbs if n in field), key=lambda n: field[n],
                          default=None)
                if nxt is None:
                    action = 6  # off-field: spin until the field is re-acquired
                else:
                    wx, wy = nxt[0] * GRID_CELL + 32, nxt[1] * GRID_CELL + 32
                    diff = (bearing_to(px, py, wx, wy) - ang + 180) % 360 - 180
                    action = pick_action(diff)
                # Wall-stuck escape: no movement for 12 steps → turn hard.
                if last_pos and math.dist(last_pos, (px, py)) < 1.0:
                    stuck += 1
                    if stuck > 12:
                        action = 6 if (stuck // 12) % 2 else 7
                else:
                    stuck = 0
                last_pos = (px, py)

                _obs, _r, term, trunc, info = env.step(action)
                if term or trunc:
                    terminal = (info.get("doom") or {}).get("terminal", "?")
                    print(f"[route] ep{ep}: terminal={terminal} at step {step} "
                          f"(closest geodesic distance reached: {best:.0f})")
                    if terminal == "exit":
                        print("[route] ✅ EXIT REACHED — the route is physically "
                              "traversable; remaining gap is pure learning.")
                        return
                    break
            else:
                print(f"[route] ep{ep}: ran out of steps "
                      f"(closest geodesic distance reached: {best:.0f})")
        print("[route] ❌ exit NOT reached — the closest-distance number above says "
              "where the next structural blocker lives.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
