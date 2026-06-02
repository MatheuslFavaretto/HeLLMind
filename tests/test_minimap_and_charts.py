"""Minimap (path) and learning-curve rendering (via cv2)."""
import os

import cv2

from writer.charts import render_learning_curve
from writer.minimap import render_minimap


def test_minimap_empty_returns_false(tmp_path):
    assert render_minimap([], os.path.join(tmp_path, "m.png")) is False


def test_minimap_writes_valid_png(tmp_path):
    cells = [[0, 0, 5], [1, 0, 2], [1, 1, 9], [2, 1, 1], [0, 2, 3]]
    out = os.path.join(tmp_path, "m.png")
    assert render_minimap(cells, out) is True
    assert os.path.exists(out)
    assert cv2.imread(out) is not None  # PNG válido


def test_minimap_draws_connected_path_line(tmp_path):
    # Only an ordered polyline (no heatmap cells, no walls) must still render.
    line = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]]
    out = os.path.join(tmp_path, "line.png")
    assert render_minimap([], out, polyline=line) is True
    assert cv2.imread(out) is not None


def test_minimap_polyline_too_short_still_ok_with_cells(tmp_path):
    # A 1-point polyline can't form a line but shouldn't crash when cells exist.
    out = os.path.join(tmp_path, "m.png")
    assert render_minimap([[0, 0, 1]], out, polyline=[[0, 0]]) is True


def test_charts_needs_two_points(tmp_path):
    one = [{"num_timesteps": 1000, "mean_reward": 1.0}]
    assert render_learning_curve(one, os.path.join(tmp_path, "c.png")) is False


def test_charts_writes_valid_png(tmp_path):
    snaps = [
        {"num_timesteps": i * 10000, "mean_reward": i * 0.5,
         "shooting_accuracy": min(1.0, i * 0.1), "kills_per_episode": i * 0.2,
         "success_rate": min(1.0, i * 0.05)}
        for i in range(1, 8)
    ]
    out = os.path.join(tmp_path, "c.png")
    assert render_learning_curve(snaps, out) is True
    assert cv2.imread(out) is not None
