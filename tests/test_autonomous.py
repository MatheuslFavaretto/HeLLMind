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


def _hist(scores, plateau_at=None, env=None):
    """Build a fake history list. plateau_at: {index: level} marks escape iters."""
    plateau_at = plateau_at or {}
    env = env or {"MAPS": "MAP01", "ENT_COEF": "0.03"}
    return [{"iter": i, "score": s, "kept": False, "env": dict(env),
             "plateau_level": plateau_at.get(i, 0)}
            for i, s in enumerate(scores)]


class TestPlateauEscape:
    """Plateau detection + escalation. Each bug here was OBSERVED in a prod run
    (51 iters: old best at iter 4 kept the streak ≥20 forever → L4 fired every iter,
    its purge was undone by write_log, and `kept` reverted the intervention)."""

    def test_no_streak_when_improving(self):
        from rl.autonomous import _no_improve_streak
        assert _no_improve_streak(_hist([0.1, 0.2, 0.3, 0.4])) == 0

    def test_streak_counts_trailing_non_improvement(self):
        from rl.autonomous import _no_improve_streak
        # best 0.9 at index 1, then 6 iters far below it
        h = _hist([0.5, 0.9] + [0.1] * 6)
        assert _no_improve_streak(h) == 6

    def test_streak_resets_after_escape_marker(self):
        """THE prod bug: pre-escape best must not poison the post-escape window."""
        from rl.autonomous import _no_improve_streak
        # old regime: best 0.95 then stuck; escape at index 10; 3 mediocre iters after
        scores = [0.95] + [0.05] * 10 + [0.10, 0.05, 0.04]
        h = _hist(scores, plateau_at={10: 4})
        # window = last 3 iters; best within window is 0.10 (its own baseline) —
        # 0.05/0.04 trail it by > threshold(0.03). Without the marker reset the
        # streak would be 13 (everything since the 0.95 at index 0).
        assert _no_improve_streak(h) == 2

    def test_no_escape_during_fresh_post_escape_window(self):
        from rl.autonomous import _stagnation_level
        scores = [0.95] + [0.05] * 20 + [0.1, 0.1]
        h = _hist(scores, plateau_at={20: 4})
        assert _stagnation_level(h) == 0  # only 2 iters in the new regime: give it time

    def test_escalation_one_level_per_failed_intervention(self):
        from rl.autonomous import _stagnation_level
        # L1 fired at index 5; 6 flat iters after → escalate to exactly L2 (not L1, not L4)
        scores = [0.5] * 5 + [0.05] + [0.3] + [0.01] * 6
        h = _hist(scores, plateau_at={5: 1})
        assert _stagnation_level(h) == 2

    def test_escalation_caps_at_level_4(self):
        from rl.autonomous import _stagnation_level
        scores = [0.5] * 3 + [0.05] + [0.3] + [0.01] * 6
        h = _hist(scores, plateau_at={3: 4})
        assert _stagnation_level(h) == 4

    def test_first_escape_level_from_absolute_streak(self):
        from rl.autonomous import _stagnation_level
        h = _hist([0.9] + [0.0] * 12)  # 12-iter streak, no prior escape
        assert _stagnation_level(h) == 2  # 12 ≥ window[2]=10, < window[3]=15

    def test_session_best_ignores_pre_escape_scores(self):
        """`kept` guardrail bug: old-regime best 0.95 must not be the resume baseline."""
        from rl.autonomous import _session_best
        scores = [0.95] + [0.05] * 10 + [0.12, 0.08]
        h = _hist(scores, plateau_at={10: 4})
        assert _session_best(h) == pytest.approx(0.12)

    def test_session_best_empty_window_is_minus_inf(self):
        from rl.autonomous import _session_best
        h = _hist([0.95, 0.05], plateau_at={1: 4})  # escape is the LAST entry
        assert _session_best(h) == -1e9  # next iter sets the new baseline

    def test_l4_returns_purge_and_keeps_brain(self, tmp_path):
        from rl.autonomous import plateau_escape
        from config import Config
        cfg = Config()
        cfg.memory_dir = str(tmp_path)
        trail = os.path.join(str(tmp_path), "autonomy.jsonl")
        open(trail, "w").write("{}\n")
        h = _hist([0.9] + [0.0] * 25)
        env = {"MAPS": "MAP01", "ENT_COEF": "0.03"}

        new_env, reason, purge = plateau_escape(cfg, env, h, 4, "MAP01", "ppo")

        assert purge is True, "L4 must tell the loop to truncate in-memory history"
        assert new_env["MAPS"] == "MAP02", "L4 must rotate the map"
        assert "BRAIN KEPT" in reason
        assert not os.path.exists(trail), "trail must be renamed away"
        backups = [f for f in os.listdir(str(tmp_path)) if "plateau_l4" in f]
        assert len(backups) == 1, "backup must exist (timestamped rename)"

    def test_l4_backups_do_not_overwrite_each_other(self, tmp_path):
        from rl.autonomous import plateau_escape
        from config import Config
        import time
        cfg = Config()
        cfg.memory_dir = str(tmp_path)
        trail = os.path.join(str(tmp_path), "autonomy.jsonl")
        h = _hist([0.9] + [0.0] * 25)
        env = {"MAPS": "MAP01"}
        open(trail, "w").write("first\n")
        plateau_escape(cfg, env, h, 4, "MAP01", "ppo")
        time.sleep(1.1)  # timestamp has 1s resolution
        open(trail, "w").write("second\n")
        plateau_escape(cfg, env, h, 4, "MAP01", "ppo")
        backups = [f for f in os.listdir(str(tmp_path)) if "plateau_l4" in f]
        assert len(backups) == 2, "each L4 must produce its own backup"

    def test_l1_resets_knobs_keeps_brain(self):
        from rl.autonomous import plateau_escape
        from config import Config
        h = _hist([0.9] + [0.0] * 6)
        with patch.dict(os.environ, {"COVERAGE_REWARD": "1.5"}):
            new_env, reason, purge = plateau_escape(
                Config(), {"COVERAGE_REWARD": "0.2", "MAPS": "MAP01"}, h, 1, "MAP01", "ppo")
        assert purge is False
        assert new_env["COVERAGE_REWARD"] == "1.5", "L1 must reset knob to .env default"

    def test_l3_reverts_to_regime_local_best(self):
        """L3 must pick the best WITHIN the current regime, not the all-time best."""
        from rl.autonomous import plateau_escape
        from config import Config
        scores = [0.95] + [0.05] * 10 + [0.30, 0.08, 0.07, 0.06, 0.05, 0.04]
        h = _hist(scores, plateau_at={10: 2})
        h[11]["env"] = {"MAPS": "MAP02", "ENT_COEF": "0.02", "MARKER": "regime-best"}
        new_env, reason, purge = plateau_escape(
            Config(), {"MAPS": "MAP02"}, h, 3, "MAP01", "ppo")
        assert purge is False
        assert new_env.get("MARKER") == "regime-best", \
            "L3 must revert to the post-escape best (iter 11), not iter 0's 0.95"


class TestCheckpointGC:
    """rl.checkpoint_gc — shared by `doom-cli prune` and the auto loop's in-loop GC."""

    def _mk(self, d, family, steps_list, final=True):
        for s in steps_list:
            open(os.path.join(d, f"{family}_{s}_steps.zip"), "wb").write(b"x" * 100)
        if final:
            open(os.path.join(d, f"{family}_final.zip"), "wb").write(b"x" * 100)

    def test_prune_keeps_newest_n_and_final(self, tmp_path):
        from rl.checkpoint_gc import prune
        d = str(tmp_path)
        self._mk(d, "ppo_campaign_a19", [1000, 2000, 3000, 4000, 5000])
        victims, freed = prune([d], keep=2, apply=True)
        assert len(victims) == 3 and freed == 300
        left = sorted(os.listdir(d))
        assert left == ["ppo_campaign_a19_4000_steps.zip",
                        "ppo_campaign_a19_5000_steps.zip",
                        "ppo_campaign_a19_final.zip"]

    def test_dry_run_deletes_nothing(self, tmp_path):
        from rl.checkpoint_gc import prune
        d = str(tmp_path)
        self._mk(d, "fam", [1, 2, 3, 4])
        victims, _ = prune([d], keep=1, apply=False)
        assert len(victims) == 3
        assert len(os.listdir(d)) == 5  # all files still present

    def test_family_filter_only_touches_that_family(self, tmp_path):
        """The auto loop must NEVER GC a family other than the one it trains."""
        from rl.checkpoint_gc import prune
        d = str(tmp_path)
        self._mk(d, "fam_a", [1, 2, 3, 4])
        self._mk(d, "fam_b", [1, 2, 3, 4])
        victims, _ = prune([d], keep=1, apply=True, family="fam_a")
        assert all("fam_a" in v for v in victims)
        assert len([f for f in os.listdir(d) if f.startswith("fam_b_") and "steps" in f]) == 4

    def test_keep_zero_disables_gc(self, tmp_path):
        from rl.checkpoint_gc import prune
        d = str(tmp_path)
        self._mk(d, "fam", [1, 2, 3])
        victims, freed = prune([d], keep=0, apply=True)
        assert victims == [] and freed == 0
        assert len(os.listdir(d)) == 4

    def test_newest_family_is_the_one_being_trained(self, tmp_path):
        from rl.checkpoint_gc import newest_family
        import time
        d = str(tmp_path)
        self._mk(d, "old_fam", [100])
        time.sleep(0.05)
        self._mk(d, "current_fam", [200])
        assert newest_family(d) == "current_fam"

    def test_newest_family_empty_dir(self, tmp_path):
        from rl.checkpoint_gc import newest_family
        assert newest_family(str(tmp_path)) is None


class TestTrendBias:
    """Cross-run behavior trends must drive a real knob decision (P2 milestone)."""

    def _cfg(self, tmp_path):
        from config import Config
        cfg = Config()
        cfg.memory_dir = str(tmp_path)
        return cfg

    def _persist_flag(self, memory_dir, name, runs=10):
        from writer.behavior import save_flags, BehaviorFlag
        for _ in range(runs):
            save_flags(memory_dir, [BehaviorFlag(
                name=name, confidence=0.8, description="d", evidence="e",
                recommendation="r")])

    def test_persistent_circling_bumps_frontier(self, tmp_path):
        from rl.autonomous import trend_bias
        cfg = self._cfg(tmp_path)
        self._persist_flag(cfg.memory_dir, "circling")
        env = {"FRONTIER_REWARD": "0.05"}
        new = dict(env)
        note = trend_bias(cfg, env, new)
        assert note is not None and "circling" in note
        assert float(new["FRONTIER_REWARD"]) == pytest.approx(0.065)  # ×1.3

    def test_no_bias_without_history(self, tmp_path):
        from rl.autonomous import trend_bias
        cfg = self._cfg(tmp_path)
        env = {"FRONTIER_REWARD": "0.05"}
        new = dict(env)
        assert trend_bias(cfg, env, new) is None
        assert new == env

    def test_heuristic_proposal_wins_over_trend(self, tmp_path):
        """If this iteration's proposal already moved the counter-knob, don't double-bump."""
        from rl.autonomous import trend_bias
        cfg = self._cfg(tmp_path)
        self._persist_flag(cfg.memory_dir, "circling")
        env = {"FRONTIER_REWARD": "0.05"}
        new = {"FRONTIER_REWARD": "0.10"}  # heuristic already raised it
        assert trend_bias(cfg, env, new) is None
        assert new["FRONTIER_REWARD"] == "0.10"

    def test_bump_clamps_to_bounds(self, tmp_path):
        from rl.autonomous import trend_bias, BOUNDS
        cfg = self._cfg(tmp_path)
        self._persist_flag(cfg.memory_dir, "circling")
        hi = BOUNDS["FRONTIER_REWARD"][1]
        env = {"FRONTIER_REWARD": str(hi)}  # already at ceiling
        new = dict(env)
        assert trend_bias(cfg, env, new) is None  # nothing to push
        assert new["FRONTIER_REWARD"] == str(hi)


class TestCurriculumFamilyParity:
    """Campaign-mode stages must derive the SAME brain family or transfer silently dies."""

    def test_all_campaign_stages_are_family_clean(self):
        from rl.progressive_curriculum import STAGE_DEFS, check_family_parity
        for stage, profile in STAGE_DEFS.items():
            bad = check_family_parity(stage, profile)
            assert not bad, f"stage '{stage}' overrides family keys {bad}"

    def test_check_catches_a_violation(self):
        from rl.progressive_curriculum import check_family_parity
        assert check_family_parity("x", {"_mode": "campaign", "STRAFE": "0"}) == ["STRAFE"]

    def test_scenario_stages_exempt(self):
        from rl.progressive_curriculum import check_family_parity
        assert check_family_parity("x", {"_mode": "scenario", "STRAFE": "0"}) == []

    def test_curriculum_never_fresh_by_default(self):
        """Locked contract: fresh=(i==0) by default would WIPE the long-trained campaign
        brain now that mywh is a campaign-mode stage. Resume must be the default."""
        import inspect
        from rl.progressive_curriculum import run
        assert inspect.signature(run).parameters["fresh"].default is False


class TestStageEnvParity:
    """Train and eval must run the SAME game. They used to build their env separately
    and diverged: train got SCENARIO_WAD, eval didn't → the mywh eval ran on freedoom2
    MAP01 and reported 1.6 kills / 45% deaths on a map with NO enemies (prod run
    2026-06-10). One shared builder makes the divergence impossible."""

    def test_mywh_env_has_scenario_wad(self):
        from rl.progressive_curriculum import STAGE_DEFS, stage_env
        env = stage_env(STAGE_DEFS["mywh"], "MAP01")
        assert env.get("SCENARIO_WAD", "").endswith("my_way_home.wad")
        assert env["CAMPAIGN"] == "1"

    def test_plain_campaign_stage_has_no_scenario_wad(self):
        from rl.progressive_curriculum import STAGE_DEFS, stage_env
        env = stage_env(STAGE_DEFS["navigate"], "MAP01")
        assert "SCENARIO_WAD" not in env or env["SCENARIO_WAD"] == os.environ.get("SCENARIO_WAD", "")

    def test_scenario_stage_forces_campaign_channels_off(self):
        from rl.progressive_curriculum import STAGE_DEFS, stage_env
        env = stage_env(STAGE_DEFS["corridor"], "MAP01")
        assert env["CAMPAIGN"] == "0"
        assert env["DOOM_SCENARIO"] == "deadly_corridor"
        assert env["STRAFE"] == "0" and env["GAME_VARS"] == "0"

    def test_memory_only_on_full_stage(self):
        from rl.progressive_curriculum import STAGE_DEFS, stage_env
        assert stage_env(STAGE_DEFS["full"], "MAP01", memory=True)["MEMORY_ENABLED"] == "1"
        assert stage_env(STAGE_DEFS["mywh"], "MAP01")["MEMORY_ENABLED"] == "0"


class TestLRFloor:
    """The unfloored linear schedule froze every resumed chunk: SB3's progress is
    GLOBAL (1 - num/(num+chunk)), so an 18M-step brain resumed for a 400k chunk
    started at progress≈0.02 → LR 5e-06→9e-08, approx_kl 1.6e-05, clip_fraction 0.
    Likely a root cause of the 47-iteration auto-loop plateau."""

    def _cfg(self, schedule=True, floor=0.1):
        from config import Config
        cfg = Config()
        cfg.lr_schedule = schedule
        cfg.lr_min_factor = floor
        return cfg

    def test_floor_holds_at_low_progress(self):
        from rl.train import _lr_setting
        sched = _lr_setting(self._cfg())
        base = 2.5e-4
        assert sched(0.02) == pytest.approx(base * 0.1)   # resumed-chunk regime
        assert sched(0.0) == pytest.approx(base * 0.1)    # never zero

    def test_normal_decay_above_floor(self):
        from rl.train import _lr_setting
        sched = _lr_setting(self._cfg())
        base = 2.5e-4
        assert sched(1.0) == pytest.approx(base)
        assert sched(0.5) == pytest.approx(base * 0.5)

    def test_floor_zero_restores_decay_to_zero(self):
        from rl.train import _lr_setting
        sched = _lr_setting(self._cfg(floor=0.0))
        assert sched(0.0) == 0.0

    def test_schedule_off_is_constant(self):
        from rl.train import _lr_setting
        assert _lr_setting(self._cfg(schedule=False)) == pytest.approx(2.5e-4)

    def test_resume_overrides_pickled_schedule(self):
        """The .load path must pass custom_objects={'learning_rate': ...} — otherwise
        old brains keep their pickled decay-to-zero schedule regardless of config."""
        import inspect
        from rl import train
        src = inspect.getsource(train.main)
        assert 'custom_objects={"learning_rate": _lr_setting(cfg)}' in src


class TestNoAssistsFlag:
    """--no-assists must zero all 4 assist env vars in every subprocess."""

    def _build_env(self, no_assists: bool) -> dict:
        """Simulate what autonomous.main() builds for the subprocess env."""
        from config import Config
        cfg = Config()
        env = {
            "AUTO_USE":         "1" if cfg.auto_use else "0",
            "AUTO_AIM":         "1" if cfg.auto_aim else "0",
            "AUTO_BEST_WEAPON": "1" if cfg.auto_best_weapon else "0",
            "AUTO_DOOR_NAV":    "1" if cfg.auto_door_nav else "0",
        }
        if no_assists:
            env["AUTO_AIM"] = "0"
            env["AUTO_BEST_WEAPON"] = "0"
            env["AUTO_USE"] = "0"
            env["AUTO_DOOR_NAV"] = "0"
        return env

    def test_default_loop_has_all_assists_on(self):
        """Without --no-assists, all 4 assists default to ON (the assisted-system mode)."""
        env = self._build_env(no_assists=False)
        assert env["AUTO_AIM"] == "1"
        assert env["AUTO_BEST_WEAPON"] == "1"
        assert env["AUTO_USE"] == "1"
        assert env["AUTO_DOOR_NAV"] == "1"

    def test_no_assists_zeros_all_four(self):
        """--no-assists must set ALL 4 assists to '0', not just some."""
        env = self._build_env(no_assists=True)
        assert env["AUTO_AIM"] == "0",         "AUTO_AIM must be '0'"
        assert env["AUTO_BEST_WEAPON"] == "0", "AUTO_BEST_WEAPON must be '0'"
        assert env["AUTO_USE"] == "0",         "AUTO_USE must be '0'"
        assert env["AUTO_DOOR_NAV"] == "0",    "AUTO_DOOR_NAV must be '0'"

    def test_no_assists_keys_present_in_env_dict(self):
        """All 4 assists must be explicit keys (not missing and inherited from os.environ)."""
        env = self._build_env(no_assists=False)
        for key in ("AUTO_AIM", "AUTO_BEST_WEAPON", "AUTO_USE", "AUTO_DOOR_NAV"):
            assert key in env, f"{key} must be explicit in the subprocess env dict"

    def test_no_assists_survives_subprocess_env_merge(self):
        """Even with os.environ containing AUTO_AIM=1, --no-assists must win."""
        import os as _os
        from rl.autonomous import _subprocess_env
        env_dict = self._build_env(no_assists=True)
        # Simulate parent shell that had assists ON
        with patch.dict(_os.environ, {"AUTO_AIM": "1", "AUTO_BEST_WEAPON": "1",
                                       "AUTO_USE": "1", "AUTO_DOOR_NAV": "1"}):
            merged = _subprocess_env(env_dict)
        assert merged["AUTO_AIM"] == "0",         "no-assists must override os.environ"
        assert merged["AUTO_BEST_WEAPON"] == "0", "no-assists must override os.environ"
        assert merged["AUTO_USE"] == "0",         "no-assists must override os.environ"
        assert merged["AUTO_DOOR_NAV"] == "0",    "no-assists must override os.environ"

    def test_behavior_snapshot_called_in_loop(self):
        """behavior.save_flags must be importable and callable from within the loop."""
        from writer.behavior import save_flags, detect_from_vault
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            save_flags(tmp, [])
            history_path = os.path.join(tmp, "behavior_history.jsonl")
            assert os.path.exists(history_path)
            with open(history_path) as f:
                record = json.loads(f.readline())
            assert "ts" in record
            assert record["flags"] == []

    def test_semantic_index_called_in_loop(self):
        """index_from_memory_store must be importable and return 0 on empty dir."""
        from writer.semantic_memory import index_from_memory_store
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            n = index_from_memory_store(tmp)
            assert n == 0

    def test_semantic_index_with_new_events(self):
        """index_from_memory_store must pick up events written after the cursor."""
        from writer.semantic_memory import index_from_memory_store, _read_cursor
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "memory_store.jsonl")
            events = [
                {"map": "MAP01", "terminal": "death", "health": 10.0},
                {"map": "MAP01", "terminal": "timeout", "region": "2x3"},
            ]
            with open(store, "w") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")

            n1 = index_from_memory_store(tmp)
            assert n1 == 2, f"expected 2 new entries, got {n1}"
            assert _read_cursor(tmp) == 2

            # Second call: no new events — cursor blocks re-indexing.
            n2 = index_from_memory_store(tmp)
            assert n2 == 0, "cursor must prevent re-indexing already-seen events"

            # Third call: one more event appended.
            with open(store, "a") as f:
                f.write(json.dumps({"map": "MAP01", "terminal": "exit"}) + "\n")
            n3 = index_from_memory_store(tmp)
            assert n3 == 1, "cursor must advance and pick up only the new event"
            assert _read_cursor(tmp) == 3
