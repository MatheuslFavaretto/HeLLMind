"""Prompts: o fact-sheet (não JSON cru) e os indicadores de tendência."""
from writer.prompts import _trend, build_checkpoint_user_message


def test_trend_up_down_stable_first():
    up = _trend(0.31, 0.18, pct=True)
    assert "↑" in up and "31%" in up and "18%" in up
    down = _trend(10.0, 20.0)
    assert "↓" in down
    assert "estável" in _trend(5.0, 5.0)
    assert "1º checkpoint" in _trend(3.0, None)


def _snapshot():
    return {
        "num_timesteps": 50000, "episodes": 8, "steps_in_window": 50000,
        "mean_reward": 12.0, "kills_per_episode": 3.5, "mean_episode_length": 220,
        "shots_fired": 40, "shots_hit": 12, "shots_missed": 28,
        "shooting_accuracy": 0.30, "damage_dealt": 200, "damage_taken": 50,
        "distance_per_episode": 1500, "cells_visited": 18,
        "map_coverage": {"explored_fraction": 0.42},
        "weapons_used": {"slot_2": 0.8, "slot_3": 0.2},
        "action_entropy_normalized": 0.65,
        "action_distribution": {"ATTACK": 0.5, "MOVE_FORWARD": 0.3, "TURN_LEFT": 0.2},
        "mean_health": 70, "mean_ammo": 30, "success_rate": 0.25, "map": "MAP01",
    }


def test_factsheet_is_readable_not_raw_json():
    msg = build_checkpoint_user_message(
        _snapshot(), previous=None,
        existing_concepts=["Reward Shaping"], button_names=["ATTACK"],
    )
    # Seções legíveis em vez de um dump JSON
    assert "Pontaria" in msg and "Exploração" in msg
    assert "Reward Shaping" in msg          # conceitos existentes injetados
    assert "30%" in msg                     # precisão formatada
    assert not msg.strip().startswith("{")  # não é JSON cru


def test_factsheet_shows_deltas_vs_previous():
    cur = _snapshot()
    prev = {**cur, "shooting_accuracy": 0.15}
    msg = build_checkpoint_user_message(cur, prev, [], ["ATTACK"])
    assert "↑" in msg  # precisão subiu de 15% -> 30%
