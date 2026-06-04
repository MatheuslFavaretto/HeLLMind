"""Tests for the visual overlay renderer (doom/overlay.py).
All tests are pure (no ViZDoom) — they operate on numpy arrays."""
import numpy as np
import pytest


try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

skip_no_cv2 = pytest.mark.skipif(not _CV2, reason="opencv-python not installed")


@skip_no_cv2
def test_draw_hud_modifies_bottom_of_frame():
    from doom.overlay import draw_hud
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_hud(img.copy(), health=0.7, ammo=0.4)
    # bottom rows should have non-zero pixels (bars were drawn)
    assert out[-15:, :, :].sum() > 0


@skip_no_cv2
def test_draw_hud_handles_clamp():
    from doom.overlay import draw_hud
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # should not raise on out-of-range values
    draw_hud(img, health=1.5, ammo=-0.1)


@skip_no_cv2
def test_draw_minimap_overlays_corner():
    from doom.overlay import draw_minimap
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    amap = np.ones((84, 84), dtype=np.uint8) * 128
    out = draw_minimap(img.copy(), amap, size=50, margin=4)
    # top-right corner should have pixels from the minimap
    assert out[4:54, 146:196, :].sum() > 0


@skip_no_cv2
def test_draw_enemy_boxes_skips_non_monsters():
    from doom.overlay import draw_enemy_boxes
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    labels = [{"object_name": "DoomPlayer", "x": 50, "y": 50, "width": 40, "height": 40}]
    out = draw_enemy_boxes(img.copy(), labels, 200, 200)
    assert out.sum() == 0  # player = not a monster, no box drawn


@skip_no_cv2
def test_draw_enemy_boxes_draws_monster():
    from doom.overlay import draw_enemy_boxes
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    labels = [{"object_name": "DoomImp", "x": 50, "y": 50, "width": 40, "height": 40}]
    out = draw_enemy_boxes(img.copy(), labels, 200, 200)
    assert out.sum() > 0  # box drawn


def test_overlay_functions_no_cv2():
    """Even without cv2, the module must be importable and return gracefully."""
    from doom.overlay import draw_hud, draw_minimap, draw_enemy_boxes
    img = np.zeros((84, 84, 3), dtype=np.uint8)
    # these should not raise even if cv2 is unavailable
    r1 = draw_hud(img, 0.5, 0.5)
    r2 = draw_minimap(img, None)
    r3 = draw_enemy_boxes(img, None, 84, 84)
    assert r1 is img or r1.shape == img.shape
