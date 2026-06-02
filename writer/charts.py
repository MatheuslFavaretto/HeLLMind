"""Learning curve: renders the key metrics' evolution over the run.

Reads the collected snapshots (same JSONL as the notes) and draws a line chart with
cv2 + numpy (no new dependency). Each series is normalized to [0,1] by its own min/max
so they fit in the same panel — to SEE whether the reward shaping and training are
heading the right way. Runs post-training, alongside the notes.
"""
import os
from typing import Dict, List, Sequence

import cv2
import numpy as np

# (chave no snapshot, rótulo na legenda, cor BGR)
SERIES = [
    ("mean_reward", "reward/ep", (90, 220, 90)),
    ("shooting_accuracy", "accuracy", (250, 180, 90)),
    ("kills_per_episode", "kills/ep", (90, 170, 250)),
    ("success_rate", "success", (220, 120, 230)),
]


def render_learning_curve(
    snapshots: Sequence[Dict], out_path: str, w: int = 780, h: int = 400
) -> bool:
    pts: List[Dict] = [s for s in snapshots if "num_timesteps" in s]
    if len(pts) < 2:
        return False

    xs = [float(s["num_timesteps"]) for s in pts]
    xmin, xmax = min(xs), max(xs)
    xspan = (xmax - xmin) or 1.0
    ml, mr, mt, mb = 64, 18, 40, 52
    pw, ph = w - ml - mr, h - mt - mb
    img = np.full((h, w, 3), 24, np.uint8)

    # grade + moldura
    for i in range(5):
        y = mt + int(ph * i / 4)
        cv2.line(img, (ml, y), (w - mr, y), (48, 48, 48), 1)
    cv2.rectangle(img, (ml, mt), (w - mr, h - mb), (90, 90, 90), 1)

    def X(v: float) -> int:
        return ml + int((v - xmin) / xspan * pw)

    for key, _label, color in SERIES:
        ys = [float(s.get(key, 0.0)) for s in pts]
        lo, hi = min(ys), max(ys)
        span = (hi - lo) or 1.0
        prev = None
        for x, yv in zip(xs, ys):
            px = X(x)
            py = mt + ph - int((yv - lo) / span * ph)
            if prev is not None:
                cv2.line(img, prev, (px, py), color, 2, cv2.LINE_AA)
            prev = (px, py)

    # title + legend + axes
    cv2.putText(img, "Learning curve (each series normalized)", (ml, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    for i, (_key, label, color) in enumerate(SERIES):
        y = mt + 16 + i * 18
        cv2.line(img, (ml + 8, y), (ml + 32, y), color, 2)
        cv2.putText(img, label, (ml + 38, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.putText(img, f"{int(xmin):,}", (ml, h - mb + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"{int(xmax):,} steps", (w - mr - 130, h - mb + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, img))


_PALETTE = [(90, 220, 90), (90, 170, 250), (250, 180, 90), (220, 120, 230), (120, 220, 220)]


def render_run_comparison(
    runs: Dict[str, Sequence[Dict]], metric: str, out_path: str,
    title: str = None, w: int = 780, h: int = 400,
) -> bool:
    """Overlay `metric` over time, one line per run (same Y scale)."""
    valid = {
        label: [s for s in snaps if "num_timesteps" in s and metric in s]
        for label, snaps in runs.items()
    }
    valid = {label: v for label, v in valid.items() if len(v) >= 2}
    if not valid:
        return False

    all_x = [float(s["num_timesteps"]) for v in valid.values() for s in v]
    all_y = [float(s[metric]) for v in valid.values() for s in v]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    ml, mr, mt, mb = 64, 18, 40, 52
    pw, ph = w - ml - mr, h - mt - mb
    img = np.full((h, w, 3), 24, np.uint8)
    for i in range(5):
        y = mt + int(ph * i / 4)
        cv2.line(img, (ml, y), (w - mr, y), (48, 48, 48), 1)
    cv2.rectangle(img, (ml, mt), (w - mr, h - mb), (90, 90, 90), 1)

    def X(v: float) -> int:
        return ml + int((v - xmin) / xspan * pw)

    def Y(v: float) -> int:
        return mt + ph - int((v - ymin) / yspan * ph)

    for i, (label, snaps) in enumerate(valid.items()):
        color = _PALETTE[i % len(_PALETTE)]
        prev = None
        for s in snaps:
            px, py = X(float(s["num_timesteps"])), Y(float(s[metric]))
            if prev is not None:
                cv2.line(img, prev, (px, py), color, 2, cv2.LINE_AA)
            prev = (px, py)
        ly = mt + 16 + i * 18
        cv2.line(img, (ml + 8, ly), (ml + 32, ly), color, 2)
        cv2.putText(img, label[:24], (ml + 38, ly + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (215, 215, 215), 1, cv2.LINE_AA)

    cv2.putText(img, title or metric, (ml, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, img))
