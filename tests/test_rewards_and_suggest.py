"""Configurable reward weights + reward-suggestion prompt (Phase 6)."""
from config import Config
from writer.prompts import build_suggest_user_message
from writer.suggest import ENV_VAR


def test_reward_weights_dict_has_all_knobs():
    w = Config().reward_weights()
    assert set(w) == {"hit_reward", "miss_penalty", "damage_taken_penalty", "death_penalty"}
    assert all(isinstance(v, float) for v in w.values())


def test_env_var_mapping_matches_knobs():
    assert set(ENV_VAR) == set(Config().reward_weights())
    assert ENV_VAR["hit_reward"] == "HIT_REWARD"


def test_suggest_message_shows_weights_and_behavior():
    weights = {"hit_reward": 1.0, "miss_penalty": 0.25,
               "damage_taken_penalty": 0.05, "death_penalty": 5.0}
    stats = {"shooting_accuracy": 0.12, "death_rate": 0.9, "deaths": 90,
             "low_hp_death_rate": 0.8, "mean_health_at_death": 14.0}
    msg = build_suggest_user_message(stats, weights)
    assert "hit_reward = 1.0" in msg
    assert "12%" in msg                 # accuracy
    assert "Low-HP deaths (<30): 80%" in msg
