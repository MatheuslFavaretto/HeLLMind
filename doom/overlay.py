"""Structured overlays for doom-cli watch (V2 Phase 3).

Renders the Doom frame with annotated overlays directly from the ViZDoom buffers
we already use: game-vars (HUD), labels (enemy bounding boxes), automap (minimap).
Everything is already coming from the env — we just draw it.

Usage:
    from doom.overlay import render_frame
    img_rgb = render_frame(state, cfg, frame_obs=np_obs)
"""
from typing import Any, Optional
import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ── Colour palette (ember theme) ──────────────────────────────────────────────
_RED    = (0,   45, 255)   # BGR for cv2 — #FF2D00 (enemy bbox)
_ORANGE = (0,  149, 255)   # #FF9500 (health bar)
_GOLD   = (0,  208, 255)   # #FFD000 (ammo bar)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (50, 205, 50)
_DKRED  = (0,   0, 140)    # dark red for bars


def _bar(img, x, y, w, h, fill: float, fg, bg=_DKRED, label: str = "") -> None:
    """Draw a filled progress bar with a label."""
    fill = max(0.0, min(1.0, fill))
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(w * fill), y + h), fg, -1)
    if label:
        cv2.putText(img, label, (x + 4, y + h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _WHITE, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, health: float, ammo: float) -> np.ndarray:
    """Draw a health bar and ammo bar (bottom of frame, non-invasive)."""
    if not _CV2:
        return img
    h, w = img.shape[:2]
    bar_h = max(10, h // 14)
    bar_w = w // 2 - 6
    margin = 4
    y = h - bar_h - margin
    _bar(img, margin, y, bar_w, bar_h,
         health, _ORANGE, label=f"HP {int(health*100)}%")
    _bar(img, w // 2 + margin, y, bar_w, bar_h,
         ammo, _GOLD, label=f"AMMO {int(ammo*100)}%")
    return img


def draw_enemy_boxes(img: np.ndarray, labels, screen_w: int, screen_h: int,
                     scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Draw bounding boxes around visible enemies from the ViZDoom labels buffer."""
    if not _CV2 or labels is None:
        return img
    from doom.entities import is_monster
    for lab in labels:
        name = getattr(lab, "object_name", None)
        if name is None and isinstance(lab, dict):
            name = lab.get("object_name")
        if not name or not is_monster(name):
            continue
        x = getattr(lab, "x", None) or lab.get("x", 0)
        y = getattr(lab, "y", None) or lab.get("y", 0)
        lw = getattr(lab, "width", None) or lab.get("width", 0)
        lh = getattr(lab, "height", None) or lab.get("height", 0)
        x1 = int(x * scale_x); y1 = int(y * scale_y)
        x2 = int((x + lw) * scale_x); y2 = int((y + lh) * scale_y)
        cv2.rectangle(img, (x1, y1), (x2, y2), _RED, 2)
        short = name.replace("Doom", "").replace("Guy", "")
        cv2.putText(img, short, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _RED, 1, cv2.LINE_AA)
    return img


def draw_minimap(img: np.ndarray, automap: np.ndarray,
                 size: int = 96, margin: int = 4) -> np.ndarray:
    """Overlay the automap (explored top-down map) in the top-right corner."""
    if not _CV2 or automap is None:
        return img
    h, w = img.shape[:2]
    # Resize automap to a small square with a dark border
    mini = cv2.resize(automap, (size, size), interpolation=cv2.INTER_NEAREST)
    if mini.ndim == 2:
        mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
    # dark border
    cv2.rectangle(mini, (0, 0), (size - 1, size - 1), _BLACK, 1)
    x0 = w - size - margin
    img[margin: margin + size, x0: x0 + size] = mini
    cv2.putText(img, "MAP", (x0, margin + size + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, _GOLD, 1, cv2.LINE_AA)
    return img


def render_frame(state: Any, cfg, frame_obs: Optional[np.ndarray] = None,
                 render_size: tuple = (420, 420)) -> Optional[np.ndarray]:
    """Compose a full annotated frame for doom-cli watch --overlay.

    Returns an RGB numpy array ready for cv2.imshow, or None if cv2 is missing.
    """
    if not _CV2:
        return None

    # ── Base frame ──
    if state is not None and hasattr(state, "screen_buffer"):
        base = state.screen_buffer.copy()
        if base.ndim == 2:                           # GRAY
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif frame_obs is not None:
        # obs tensor: take the first channel (pixels), upscale to render_size
        arr = np.asarray(frame_obs)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        base = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        return None

    orig_h, orig_w = base.shape[:2]
    rw, rh = render_size
    base = cv2.resize(base, (rw, rh), interpolation=cv2.INTER_NEAREST)
    sx, sy = rw / orig_w, rh / orig_h

    # ── HUD (health/ammo from game-vars) ──
    if state is not None and hasattr(state, "game_variables"):
        gv = {i: float(v) for i, v in enumerate(state.game_variables)}
        health_n = min(1.0, max(0.0, gv.get(0, 0.0) / 100.0))
        ammo_n   = min(1.0, max(0.0, gv.get(1, 0.0) / 50.0))
        draw_hud(base, health_n, ammo_n)

    # ── Enemy bounding boxes (labels buffer) ──
    if state is not None and hasattr(state, "labels") and state.labels:
        draw_enemy_boxes(base, state.labels, orig_w, orig_h, sx, sy)

    # ── Minimap (automap buffer) ──
    if state is not None and hasattr(state, "automap_buffer"):
        amap = getattr(state, "automap_buffer", None)
        if amap is not None:
            draw_minimap(base, amap, size=min(100, rh // 4))

    return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
