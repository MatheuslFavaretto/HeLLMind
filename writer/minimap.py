"""Level minimap: draws the real map WALLS + the path taken.

It used to be just "loose squares". Now, when the env provides the map geometry
(`map_walls`, via ViZDoom sectors), we draw the real level outline and paint a heatmap
of where the agent walked on top — all in the same world coordinates. Without geometry
(or without walls), it falls back to the old mode (cells only). Uses numpy + cv2.
"""
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np

from instrumentation.stats_tracker import COVERAGE_CELL


def _bounds(path_cells, walls, line=()):
    xs: List[float] = []
    ys: List[float] = []
    for gx, gy, _c in path_cells:
        xs.append(gx * COVERAGE_CELL)
        ys.append(gy * COVERAGE_CELL)
    for x1, y1, x2, y2 in walls:
        xs += [x1, x2]
        ys += [y1, y2]
    for gx, gy in line:
        xs.append(gx * COVERAGE_CELL)
        ys.append(gy * COVERAGE_CELL)
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def render_minimap(
    path_cells: Sequence[Sequence[float]],
    out_path: str,
    walls: Optional[Sequence[Sequence[float]]] = None,
    polyline: Optional[Sequence[Sequence[float]]] = None,
    memory_cells: Optional[Sequence[Sequence[float]]] = None,
    target_px: int = 420,
    pad: int = 12,
) -> bool:
    """Generate the minimap PNG (real walls + path). True if the file was written.

    Layers, bottom to top: level walls; the persistent cross-run exploration memory
    (`memory_cells`, faint cool fill — everywhere the agent has EVER been on this map);
    this run's visit heatmap (`path_cells`, hot); and — when an ordered trajectory is
    given (`polyline`, [gx, gy] in visit order) — the path as a CONNECTED LINE with
    start (green) and end (red) markers.
    """
    cells: List[Sequence[float]] = [c for c in (path_cells or []) if c]
    walls = [w for w in (walls or []) if w and len(w) == 4]
    line: List[Sequence[float]] = [p for p in (polyline or []) if p and len(p) >= 2]
    mem: List[Sequence[float]] = [c for c in (memory_cells or []) if c]
    if not cells and not walls and not line and not mem:
        return False

    b = _bounds(cells + mem, walls, [(p[0], p[1]) for p in line])
    if b is None:
        return False
    minx, maxx, miny, maxy = b
    span_x = (maxx - minx) or 1.0
    span_y = (maxy - miny) or 1.0
    scale = (target_px - 2 * pad) / max(span_x, span_y)
    w_img = int(span_x * scale) + 2 * pad
    h_img = int(span_y * scale) + 2 * pad
    img = np.full((h_img, w_img, 3), 18, np.uint8)

    def to_px(x: float, y: float):
        px = int((x - minx) * scale) + pad
        py = int((maxy - y) * scale) + pad  # Doom's Y grows upward -> invert
        return px, py

    # 1) Level walls (background).
    for x1, y1, x2, y2 in walls:
        cv2.line(img, to_px(x1, y1), to_px(x2, y2), (110, 110, 120), 1, cv2.LINE_AA)

    # 1.5) Persistent exploration MEMORY (cross-run): faint cool fill of every cell the
    # agent has ever stepped on for this map. Sits under the current run's hot heatmap.
    if mem:
        half = COVERAGE_CELL / 2.0
        overlay = img.copy()
        for gx, gy, _c in mem:
            cx, cy = gx * COVERAGE_CELL, gy * COVERAGE_CELL
            p1, p2 = to_px(cx - half, cy + half), to_px(cx + half, cy - half)
            cv2.rectangle(overlay, p1, p2, (90, 60, 30), -1)  # muted blue-teal
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # 2) Path taken on top (heatmap by visit count), translucent.
    if cells:
        max_c = max(float(c[2]) for c in cells) or 1.0
        half = COVERAGE_CELL / 2.0
        overlay = img.copy()
        for gx, gy, c in cells:
            cx, cy = gx * COVERAGE_CELL, gy * COVERAGE_CELL
            inten = int(np.log1p(float(c)) / np.log1p(max_c) * 255)
            color = tuple(int(v) for v in cv2.applyColorMap(
                np.array([[inten]], np.uint8), cv2.COLORMAP_INFERNO)[0, 0])
            p1, p2 = to_px(cx - half, cy + half), to_px(cx + half, cy - half)
            cv2.rectangle(overlay, p1, p2, color, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # 3) Ordered path as a connected line (visit order) + start/end markers, on top.
    if line:
        pts = []
        for p in line:
            cx, cy = float(p[0]) * COVERAGE_CELL, float(p[1]) * COVERAGE_CELL
            pts.append(to_px(cx, cy))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], False,
                          (80, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(img, pts[0], 4, (80, 230, 80), -1, cv2.LINE_AA)   # start (green)
        cv2.circle(img, pts[-1], 4, (60, 60, 235), -1, cv2.LINE_AA)  # end (red)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, img))
