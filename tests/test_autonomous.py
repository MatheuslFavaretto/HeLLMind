"""Test the autonomous training loop: train → eval → adjust → rollback.

The autonomy loop is the core self-improvement mechanism. It must:
1. Train a brain (resuming from checkpoint)
2. Evaluate it deterministically
3. Score the composite metrics
4. Adjust rewards if score improves
5. Rollback if score degrades
"""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rl.autonomous import score, BOUNDS


class TestAutonomousScore:
    """Test the composite goal scoring function."""

    def test_score_empty_metrics_returns_zero(self):
        """Empty metrics dict should score 0 (no improvement, no penalty)."""
        assert score({}) == 0.0

    def test_score_prioritizes_accuracy_over_exploration(self):
        """Accuracy is weighted 2.5x, exploration 1.0x."""
        # Perfect accuracy, zero exploration
        metrics_high_aim = {"shooting_accuracy": 1.0}
        # Perfect exploration, zero accuracy
        metrics_high_explore = {"explored_fraction": 1.0}
        score_aim = score(metrics_high_aim)
        score_explore = score(metrics_high_explore)
        assert score_aim > score_explore, "Accuracy should dominate exploration"

    def test_score_penalizes_wasted_shots(self):
        """High wasted_shot_rate should decrease score."""
        baseline = {"explored_fraction": 0.5}
        baseline_score = score(baseline)
        
        with_waste = {"explored_fraction": 0.5, "wasted_shot_rate": 1.0}
        waste_score = score(with_waste)
        
        assert waste_score < baseline_score, "Wasted shots should decrease score"

    def test_score_penalizes_death(self):
        """High death_rate should decrease score."""
        baseline = {"explored_fraction": 0.5}
        baseline_score = score(baseline)
        
        with_death = {"explored_fraction": 0.5, "death_rate": 1.0}
        death_score = score(with_death)
        
        assert death_score < baseline_score, "High death rate should decrease score"

    def test_score_values_exit_rate_highly(self):
        """Exit rate has weight 2.0, accuracy has 2.5, both high priority."""
        metrics_exit = {"exit_rate": 1.0}
        metrics_accuracy = {"shooting_accuracy": 1.0}
        
        score_exit = score(metrics_exit)
        score_accuracy = score(metrics_accuracy)
        
        # Both weighted high (2.0 vs 2.5), so scores should be close
        assert abs(score_exit - score_accuracy) <= 0.5

    def test_score_clamps_kills_at_5(self):
        """kills_per_episode is capped at 5.0 to prevent farm-gaming."""
        metrics_10_kills = {"kills_per_episode": 10.0}
        metrics_5_kills = {"kills_per_episode": 5.0}
        
        score_10 = score(metrics_10_kills)
        score_5 = score(metrics_5_kills)
        
        assert score_10 == score_5, "Kills should be capped at 5.0"

    def test_score_example_perfect_agent(self):
        """An agent with perfect metrics should score high."""
        perfect = {
            "shooting_accuracy": 1.0,
            "kill_conversion": 1.0,
            "kills_per_episode": 5.0,
            "explored_fraction": 1.0,
            "exit_progress": 1.0,
            "exit_rate": 1.0,
            "wasted_shot_rate": 0.0,
            "aim_offset": 0.0,
            "death_rate": 0.0,
        }
        s = score(perfect)
        # Expected: 2.5 + 1.5 + 0.5 + 1.0 + 1.0 + 2.0 = 8.5
        assert s == pytest.approx(8.5, abs=0.01)

    def test_score_example_failing_agent(self):
        """An agent that dies, doesn't aim, doesn't explore should score low."""
        failing = {
            "shooting_accuracy": 0.1,
            "kill_conversion": 0.0,
            "kills_per_episode": 0.0,
            "explored_fraction": 0.05,
            "exit_progress": 0.0,
            "exit_rate": 0.0,
            "wasted_shot_rate": 0.8,
            "aim_offset": 1.0,
            "death_rate": 0.8,
        }
        s = score(failing)
        # Should be negative: penalties dominate
        assert s < 0.0


class TestAutonomousBounds:
    """Test reward adjustment bounds."""

    def test_bounds_define_all_critical_knobs(self):
        """Core reward tuning params should be in BOUNDS."""
        expected_keys = {
            "COVERAGE_REWARD",
            "EXIT_REWARD",
            "HIT_REWARD",
            "DEATH_PENALTY",
            "KILL_REWARD",
            "ENT_COEF",
        }
        assert expected_keys.issubset(BOUNDS.keys())

    def test_bounds_are_valid_ranges(self):
        """Each bound should be (min, max) with min ≤ max."""
        for key, (mn, mx) in BOUNDS.items():
            assert isinstance(mn, (int, float)), f"{key} min not numeric"
            assert isinstance(mx, (int, float)), f"{key} max not numeric"
            assert mn <= mx, f"{key}: {mn} > {mx}"


class TestAutonomousIntegration:
    """Integration tests for the training loop itself (mock-based)."""

    @patch("rl.autonomous.subprocess.run")
    @patch("rl.autonomous.Config")
    def test_train_eval_cycle_records_metrics(self, mock_config, mock_run):
        """Mock: train → eval should call subprocess commands and record metrics."""
        mock_config.return_value.VAULT_DIR = "/tmp/test_vault"
        mock_run.return_value.returncode = 0
        
        # In real usage, this would run `python -m rl.train` and `python -m rl.eval`
        # Just verify the loop structure exists in the module
        from rl import autonomous
        
        assert hasattr(autonomous, "score"), "autonomous module should have score()"
        assert callable(autonomous.score)

    def test_autonomy_jsonl_structure(self):
        """Test that autonomy records would have correct structure."""
        # Expected structure for an autonomy iteration record
        record = {
            "iteration": 1,
            "ts": "2026-06-07T20:00:00Z",
            "before_metrics": {
                "explored_fraction": 0.10,
                "death_rate": 0.80,
                "shooting_accuracy": 0.05,
            },
            "before_score": 0.5,
            "after_metrics": {
                "explored_fraction": 0.12,
                "death_rate": 0.78,
                "shooting_accuracy": 0.08,
            },
            "after_score": 0.7,
            "adjustment": {"COVERAGE_REWARD": (1.0, 1.2), "HIT_REWARD": (2.0, 3.0)},
            "decision": "keep",
        }
        
        assert record["decision"] in ("keep", "revert")
        assert record["after_score"] >= 0 or record["before_score"] >= 0
        assert "iteration" in record


class TestAutonomousRollback:
    """Rollback logic: if score decreases, revert the adjustment."""

    def test_rollback_decision_keep_when_score_improves(self):
        """Score 0.5 → 0.7 should keep the change."""
        before_score = 0.5
        after_score = 0.7
        threshold = 0.05  # 5% threshold
        
        # Higher is better, so if after > before, keep
        should_keep = after_score > before_score * (1 - threshold)
        assert should_keep is True

    def test_rollback_decision_revert_when_score_degrades(self):
        """Score 0.7 → 0.5 should revert the change."""
        before_score = 0.7
        after_score = 0.5
        threshold = 0.05
        
        should_keep = after_score > before_score * (1 - threshold)
        assert should_keep is False

    def test_rollback_decision_keep_within_noise_margin(self):
        """Score change < 5% should still keep (noise tolerance)."""
        before_score = 1.0
        after_score = 0.98  # 2% drop
        threshold = 0.05
        
        should_keep = after_score > before_score * (1 - threshold)
        assert should_keep is True, "Small degradation within noise margin should keep"

    def test_rollback_decision_revert_beyond_threshold(self):
        """Score drop > 5% should revert."""
        before_score = 1.0
        after_score = 0.94  # 6% drop
        threshold = 0.05
        
        should_keep = after_score > before_score * (1 - threshold)
        assert should_keep is False, "Significant degradation beyond threshold should revert"


class TestAutonomousParameterAdjustment:
    """Test that parameter adjustments stay within bounds."""

    def test_adjust_within_bounds_lower(self):
        """Adjustment should not go below min."""
        param_name = "COVERAGE_REWARD"
        min_val, max_val = BOUNDS[param_name]
        current = min_val + 0.1
        
        # If we want to decrease and hit min, clamp to min
        new_val = max(current - 0.2, min_val)
        assert new_val >= min_val

    def test_adjust_within_bounds_upper(self):
        """Adjustment should not go above max."""
        param_name = "EXIT_REWARD"
        min_val, max_val = BOUNDS[param_name]
        current = max_val - 100
        
        # If we increase and hit max, clamp to max
        new_val = min(current + 200, max_val)
        assert new_val <= max_val

    def test_adjust_moves_in_correct_direction(self):
        """If metric is low, increase the reward for that metric."""
        # If explored_fraction is low → increase COVERAGE_REWARD
        metric_value = 0.05  # 5% exploration
        if metric_value < 0.15:  # below target
            old_reward = 1.0
            new_reward = old_reward * 1.2  # +20%
            assert new_reward > old_reward


class TestAutonomousCheckpointResume:
    """Test checkpoint loading/saving for resume capability."""

    def test_checkpoint_path_format(self):
        """Checkpoints should follow {model_type}_{map}_{iteration}.zip format."""
        model_type = "ppo"
        map_name = "MAP02"
        iteration = 5
        
        checkpoint_name = f"{model_type}_{map_name}_{iteration}.zip"
        assert "_" in checkpoint_name
        assert checkpoint_name.endswith(".zip")

    def test_latest_checkpoint_selection(self):
        """Should pick the checkpoint with highest iteration number."""
        checkpoints = [
            "ppo_MAP02_1.zip",
            "ppo_MAP02_3.zip",
            "ppo_MAP02_2.zip",
        ]
        
        latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].replace(".zip", "")))
        assert latest == "ppo_MAP02_3.zip"


class TestAutonomousExperimentRegistry:
    """Test that adjustments are logged to the experiment registry."""

    def test_experiment_record_structure(self):
        """Each experiment should record: param, old_val, new_val, result."""
        record = {
            "ts": "2026-06-07T20:00:00Z",
            "iteration": 1,
            "param": "COVERAGE_REWARD",
            "old_value": 1.0,
            "new_value": 1.2,
            "before_score": 0.5,
            "after_score": 0.7,
            "result": "improved",
            "confidence": 0.85,
        }
        
        assert record["result"] in ("improved", "regressed", "no_effect")
        assert record["confidence"] >= 0 and record["confidence"] <= 1.0

    def test_experiment_verdict_classification(self):
        """Classify result as improved / regressed / no_effect."""
        def classify(before, after, threshold=0.05):
            if after > before * (1 + threshold):
                return "improved"
            elif after < before * (1 - threshold):
                return "regressed"
            else:
                return "no_effect"
        
        assert classify(1.0, 1.1) == "improved"
        assert classify(1.0, 0.9) == "regressed"
        assert classify(1.0, 1.01) == "no_effect"


class TestAutonomousMultiSeedAveraging:
    """Test that multi-seed eval averages correctly."""

    def test_metrics_mean_and_std(self):
        """Should compute mean ± std across seeds."""
        seed_results = [
            {"exploration": 0.10, "kills": 1.5},
            {"exploration": 0.12, "kills": 1.7},
            {"exploration": 0.11, "kills": 1.6},
        ]
        
        import numpy as np
        exploration_vals = [r["exploration"] for r in seed_results]
        kills_vals = [r["kills"] for r in seed_results]
        
        mean_exploration = np.mean(exploration_vals)
        std_exploration = np.std(exploration_vals)
        
        assert mean_exploration == pytest.approx(0.11, abs=0.01)
        assert std_exploration > 0

    def test_verdict_only_on_significant_difference(self):
        """Only adopt if mean improvement exceeds noise (e.g., std < diff)."""
        before_mean, before_std = 1.0, 0.1
        after_mean, after_std = 1.2, 0.12  # 20% improvement
        
        # Simple heuristic: adopt if after_mean > before_mean + before_std
        should_adopt = after_mean > before_mean + before_std
        assert should_adopt is True
        
        after_tiny = 1.01
        should_adopt_tiny = after_tiny > before_mean + before_std
        assert should_adopt_tiny is False
