"""Detecção de regressão (D) e resumo/gráfico de comparação de runs (B)."""
import os

import cv2

from writer.analysis import detect_regressions
from writer.charts import render_run_comparison
from writer.compare_runs import summarize


# ----------------------------- D: regressão -----------------------------
def test_no_previous_means_no_regression():
    assert detect_regressions({"shooting_accuracy": 0.4}, None) == []


def test_detects_sharp_drop():
    cur = {"shooting_accuracy": 0.15, "mean_reward": 9.0}
    prev = {"shooting_accuracy": 0.40, "mean_reward": 10.0}
    regs = detect_regressions(cur, prev)
    # precisão caiu 0.40 -> 0.15 (−63%) -> regressão; recompensa caiu só 10% -> não
    assert any("precisão de tiro" in r for r in regs)
    assert not any("recompensa" in r for r in regs)


def test_small_drop_not_flagged():
    assert detect_regressions({"kills_per_episode": 4.5}, {"kills_per_episode": 5.0}) == []


def test_zero_baseline_skipped():
    # sem base positiva não dá pra falar em "queda"
    assert detect_regressions({"success_rate": 0.0}, {"success_rate": 0.0}) == []


# --------------------------- B: comparação ---------------------------
def _run(rewards):
    return [{"num_timesteps": i * 10000, "mean_reward": r,
             "shooting_accuracy": min(1.0, 0.1 * i), "kills_per_episode": r / 2,
             "success_rate": 0.0, "distance_per_episode": 100 * i}
            for i, r in enumerate(rewards, 1)]


def test_summarize_final_best_mean():
    s = summarize(_run([2.0, 6.0, 4.0]))
    assert s["checkpoints"] == 3
    assert s["mean_reward"]["final"] == 4.0
    assert s["mean_reward"]["best"] == 6.0
    assert s["mean_reward"]["mean"] == 4.0


def test_summarize_empty():
    s = summarize([])
    assert s["checkpoints"] == 0 and s["mean_reward"]["final"] == 0.0


def test_comparison_chart_renders(tmp_path):
    runs = {"A": _run([1, 2, 3, 4]), "B": _run([2, 3, 1, 5])}
    out = os.path.join(tmp_path, "cmp.png")
    assert render_run_comparison(runs, "mean_reward", out, title="Recompensa") is True
    assert cv2.imread(out) is not None


def test_comparison_chart_needs_data(tmp_path):
    runs = {"A": [{"num_timesteps": 1, "mean_reward": 1.0}]}  # só 1 ponto
    assert render_run_comparison(runs, "mean_reward", os.path.join(tmp_path, "x.png")) is False
