"""Tests for writer.html_report — the post-run HTML report (pure render)."""
from writer.html_report import render_html


def _metrics():
    return {"exit_progress": 0.35, "shooting_accuracy": 0.13, "kills_per_episode": 10.3,
            "aim_offset": 0.93, "wasted_shot_rate": 0.72, "revisit_rate": 0.94, "episodes": 10,
            "map_coverage": {"explored_fraction": 0.25, "cells_visited": 235},
            "reward_breakdown": {"combat": 0.07, "explore": 0.82}}


def test_renders_sections_and_numbers():
    h = render_html(_metrics(), meta={"map": "MAP01"})
    for section in ("AIM", "MOVEMENT", "SURVIVAL", "Recommendations", "Formulas"):
        assert section in h
    assert "MAP01" in h
    assert "72%" in h            # wasted_shot_rate formatted


def test_recommendations_flag_the_spray():
    h = render_html(_metrics())
    low = h.lower()
    assert "wasted" in low or "spray" in low or "exploration" in low   # actionable rec present


def test_empty_metrics_is_safe():
    h = render_html({})
    assert "<html" in h and "HeLLMind" in h
