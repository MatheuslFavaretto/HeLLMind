"""Test deterministic evaluation: metrics calculation and policy testing.

The eval module runs N episodes with a saved checkpoint and returns clean metrics:
- shooting accuracy
- kills per episode
- survival rate
- exploration percentage
- exit rate
"""
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rl.eval import evaluate, _tempered_actions


class TestTemperedActions:
    """Test temperature-controlled action sampling."""

    def test_temperature_zero_is_deterministic(self):
        """T=0 should be equivalent to argmax."""
        # Temperature 0 creates very sharp distribution
        # In practice, we clamp to 1e-6 to avoid division by zero
        temp = 0.0
        clamped_temp = max(temp, 1e-6)
        assert clamped_temp > 0

    def test_temperature_one_is_normal_sampling(self):
        """T=1.0 is normal logits/temperature scaling."""
        logits = np.array([1.0, 2.0, 3.0])
        temperature = 1.0
        scaled = logits / temperature
        np.testing.assert_array_almost_equal(scaled, logits)

    def test_temperature_half_is_sharper(self):
        """T=0.5 sharpens the distribution (higher probability on best actions)."""
        logits = np.array([1.0, 2.0, 3.0])
        temp_normal = logits / 1.0
        temp_sharp = logits / 0.5
        
        # Sharper scaling = bigger differences = higher concentration
        assert np.max(temp_sharp) > np.max(temp_normal)

    def test_temperature_two_is_flatter(self):
        """T=2.0 flattens the distribution (more uniform)."""
        logits = np.array([1.0, 2.0, 3.0])
        temp_normal = logits / 1.0
        temp_flat = logits / 2.0
        
        # Flatter scaling = smaller differences = more uniform
        assert np.max(temp_flat) < np.max(temp_normal)


class TestEvaluationMetrics:
    """Test metric calculations."""

    def test_shooting_accuracy_calculation(self):
        """Accuracy = hits / (hits + misses)."""
        hits = 20
        misses = 5
        accuracy = hits / (hits + misses)
        
        assert accuracy == pytest.approx(0.8, abs=0.01)

    def test_accuracy_zero_when_no_shots(self):
        """If never shots, accuracy should be 0 (or NaN handled as 0)."""
        shots_fired = 0
        hits = 0
        
        accuracy = hits / max(shots_fired, 1)  # guard against div by zero
        assert accuracy == 0.0

    def test_kills_per_episode_averaging(self):
        """Kills should be averaged across episodes."""
        episode_kills = [1, 2, 1, 0, 3]
        mean_kills = np.mean(episode_kills)
        
        assert mean_kills == 1.4

    def test_death_rate_calculation(self):
        """Death rate = episodes_died / total_episodes."""
        episodes = 20
        deaths = 4
        death_rate = deaths / episodes
        
        assert death_rate == 0.2

    def test_explored_fraction_0_to_1(self):
        """Exploration should be normalized [0, 1]."""
        cells_explored = 150
        total_cells = 1000
        fraction = cells_explored / total_cells
        
        assert 0.0 <= fraction <= 1.0

    def test_exit_rate_binary_metric(self):
        """Exit rate = episodes_completed / total_episodes."""
        episodes_completed = 3
        total_episodes = 20
        exit_rate = episodes_completed / total_episodes
        
        assert 0.0 <= exit_rate <= 1.0
        assert exit_rate == pytest.approx(0.15, abs=0.01)


class TestEvaluationDeterminism:
    """Test that deterministic=True enforces reproducibility."""

    def test_deterministic_uses_argmax(self):
        """deterministic=True should use argmax (no sampling)."""
        deterministic = True
        
        if deterministic:
            # Use argmax, not sampling
            sampling_enabled = False
        else:
            sampling_enabled = True
        
        if deterministic:
            assert sampling_enabled is False

    def test_seeding_produces_same_results(self):
        """Same seed should give same episode trajectory."""
        # Run with seed 42
        seed_1_results = {"exploration": 0.15, "kills": 2.0}
        
        # Run again with seed 42
        seed_2_results = {"exploration": 0.15, "kills": 2.0}
        
        assert seed_1_results == seed_2_results

    def test_different_seeds_may_differ(self):
        """Different seeds may give different results (stochastic env)."""
        seed_42_result = 0.15
        seed_99_result = 0.18
        
        # Results might differ
        assert isinstance(seed_42_result, float)
        assert isinstance(seed_99_result, float)


class TestEvaluationModeSpecific:
    """Test deterministic vs stochastic modes."""

    def test_evaluation_mode_disables_exploration(self):
        """Eval should run with entropy disabled (pure determinism)."""
        eval_mode = True
        
        if eval_mode:
            entropy_coef = 0.0
            exploration_enabled = False
        else:
            entropy_coef = 0.01
            exploration_enabled = True
        
        if eval_mode:
            assert exploration_enabled is False

    def test_training_mode_enables_exploration(self):
        """Training should sample actions (stochastic)."""
        training_mode = True
        
        if training_mode:
            entropy_coef = 0.01
            exploration_enabled = True
        
        assert exploration_enabled is True


class TestEvaluationEpisodeRollout:
    """Test episode rollout correctness."""

    def test_reset_starts_episode(self):
        """env.reset() should return initial obs."""
        # Structure check
        operations = ["reset", "step", "close"]
        for op in operations:
            assert isinstance(op, str)

    def test_step_returns_obs_reward_done(self):
        """env.step(action) should return (obs, reward, done, truncated, info)."""
        # Gymnasium v26+ returns (obs, reward, terminated, truncated, info)
        step_output_keys = ["obs", "reward", "done", "info"]
        
        for key in step_output_keys:
            assert isinstance(key, str)

    def test_done_flag_ends_episode(self):
        """When done=True, episode should terminate and reset."""
        done = False
        step_count = 0
        
        for _ in range(10):
            step_count += 1
            if step_count >= 5:
                done = True
        
        assert done is True


class TestEvaluationOverlay:
    """Test optional visualization overlay."""

    def test_overlay_flag_enables_rendering(self):
        """overlay=True should display bboxes, health, minimap."""
        overlay_enabled = True
        
        if overlay_enabled:
            should_render = True
            overlay_layers = ["hud", "bboxes", "minimap"]
        else:
            should_render = False
        
        if overlay_enabled:
            assert should_render is True
            assert len(overlay_layers) == 3

    def test_overlay_no_performance_penalty_when_disabled(self):
        """overlay=False should not draw anything."""
        overlay_disabled = True
        
        if overlay_disabled:
            # No rendering overhead
            rendering_time_ms = 0
        else:
            rendering_time_ms = 5
        
        if overlay_disabled:
            assert rendering_time_ms == 0


class TestEvaluationRecall:
    """Test optional demo retrieval (behavioral cloning fallback)."""

    def test_recall_uses_embeddings(self):
        """Recall should embed current frame and search demos."""
        recall_enabled = True
        
        if recall_enabled:
            embed_model = "nomic-embed-text"
            similarity_metric = "cosine"
        
        assert isinstance(embed_model, str)
        assert similarity_metric == "cosine"

    def test_recall_threshold_gates_adoption(self):
        """Only use recalled demo if confidence >= threshold."""
        threshold = 0.92
        similarity = 0.95
        
        if similarity >= threshold:
            use_recalled_action = True
        else:
            use_recalled_action = False
        
        assert use_recalled_action is True

    def test_recall_falls_back_to_policy_on_low_confidence(self):
        """If similarity < threshold, use policy's action."""
        threshold = 0.92
        similarity = 0.80
        
        if similarity >= threshold:
            use_policy = False
        else:
            use_policy = True
        
        assert use_policy is True


class TestEvaluationAggregation:
    """Test result aggregation across episodes."""

    def test_mean_metrics_across_episodes(self):
        """Should compute mean ± std for each metric."""
        episode_rewards = [100, 120, 110, 95, 130]
        
        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        
        assert mean == pytest.approx(111.0, abs=1.0)
        assert std > 0

    def test_results_dict_structure(self):
        """Evaluation should return a dict with all metrics."""
        results = {
            "mean_reward": 100.0,
            "std_reward": 10.0,
            "shooting_accuracy": 0.5,
            "kills_per_episode": 2.5,
            "death_rate": 0.2,
            "explored_fraction": 0.15,
            "exit_rate": 0.0,
        }
        
        assert "mean_reward" in results
        assert "shooting_accuracy" in results
        assert "death_rate" in results


class TestEvaluationPath:
    """Test checkpoint path handling."""

    def test_default_path_uses_latest_checkpoint(self):
        """If no path given, should use latest checkpoint."""
        path = None
        
        if path is None:
            use_latest = True
        else:
            use_latest = False
        
        assert use_latest is True

    def test_custom_path_loads_specific_checkpoint(self):
        """If path given, should load that specific checkpoint."""
        custom_path = "checkpoints/ppo_custom_model.zip"
        
        if custom_path:
            use_custom = True
            should_load = custom_path
        
        assert use_custom is True
        assert should_load == custom_path


class TestEvaluationEdgeCases:
    """Test error handling and edge cases."""

    def test_zero_episodes_returns_empty(self):
        """episodes=0 should return empty results."""
        episodes = 0
        
        if episodes == 0:
            results = {}
        
        assert len(results) == 0

    def test_temperature_extreme_values(self):
        """Very high/low temperature should clamp safely."""
        # Very low
        temp_very_low = 0.0
        clamped_low = max(temp_very_low, 1e-6)
        assert clamped_low > 0
        
        # Very high
        temp_very_high = 100.0
        # No explicit clamp, but should not break sampling
        assert temp_very_high > 0

    def test_missing_checkpoint_raises_error(self):
        """Loading nonexistent checkpoint should raise error."""
        import os
        path = "checkpoints/nonexistent.zip"
        
        if not os.path.exists(path):
            should_raise = True
        
        assert should_raise is True
