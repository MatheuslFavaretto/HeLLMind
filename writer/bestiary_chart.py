"""Bestiary bar chart: per-monster seen / killed-by-agent / killed-the-agent.

Grouped vertical bars (cv2 + numpy, no new dependency), one cluster per monster, so the
Obsidian note shows the combat balance at a glance: which monsters the agent meets, beats,
and dies to.
"""
import os
from typing import Any, Dict

import cv2
import numpy as np

# (store key, legend label, BGR color)
BARS = [
    ("total", "seen", (180, 180, 180)),
    ("killed", "killed by agent", (90, 200, 90)),
    ("killed_agent", "killed the agent", (70, 70, 230)),
]


def render_bestiary_chart(store: Dict[str, Any], out_path: str,
                          w: int = 780, h: int = 360) -> bool:
    """One cluster of bars per monster. True if the PNG was written."""
    from writer.bestiary import display_name
    mons = [(n, s) for n, s in store.items()
            if any(int(s.get(k, 0)) for k, _, _ in BARS)]
    if not mons:
        return False
    mons.sort(key=lambda kv: -int(kv[1].get("killed", 0)) - int(kv[1].get("total", 0)))

    ml, mr, mt, mb = 44, 16, 44, 40
    pw, ph = w - ml - mr, h - mt - mb
    img = np.full((h, w, 3), 24, np.uint8)
    vmax = max(1, max(int(s.get(k, 0)) for _, s in mons for k, _, _ in BARS))

    for i in range(5):  # gridlines
        y = mt + int(ph * i / 4)
        cv2.line(img, (ml, y), (w - mr, y), (48, 48, 48), 1)
    cv2.rectangle(img, (ml, mt), (w - mr, h - mb), (90, 90, 90), 1)

    slot = pw / len(mons)
    bw = min(26, slot / (len(BARS) + 1))
    for ci, (name, s) in enumerate(mons):
        x0 = ml + slot * ci + (slot - bw * len(BARS)) / 2
        for bi, (key, _label, color) in enumerate(BARS):
            v = int(s.get(key, 0))
            bh = int(v / vmax * ph)
            x = int(x0 + bi * bw)
            y = mt + ph - bh
            cv2.rectangle(img, (x, y), (int(x + bw - 2), mt + ph), color, -1)
            if v:
                cv2.putText(img, str(v), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (210, 210, 210), 1, cv2.LINE_AA)
        label = display_name(name)[:9]
        cv2.putText(img, label, (int(ml + slot * ci + 2), h - mb + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(img, "Bestiary - combat by monster", (ml, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    for i, (_key, lbl, color) in enumerate(BARS):  # legend
        x = w - mr - 150
        y = 16 + i * 16
        cv2.rectangle(img, (x, y - 8), (x + 12, y + 2), color, -1)
        cv2.putText(img, lbl, (x + 18, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (210, 210, 210), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, img))
