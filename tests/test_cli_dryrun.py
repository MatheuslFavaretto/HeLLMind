"""Dry-run every doom-cli entry point with subprocess/run mocked.

Why: two NameError-class bugs reached prod this week through code paths that are only
exercised when a HUMAN runs the command (cmd builders, env plumbing, title strings).
Each handler here is invoked exactly as argparse would, with the actual parser from
build_parser(), so a broken reference inside any handler fails CI instead of a run.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

import doom_cli


# command → extra argv after the subcommand name. Curated: every command whose handler
# builds a subprocess invocation (the NameError surface). Interactive/recursive ones
# (shell, clean, tests) are deliberately excluded.
DRYRUN_ARGV = {
    "train":       [],
    "dqn":         [],
    "watch":       ["--maps", "MAP02", "--temperature", "0.5"],
    "eval":        ["--maps", "MAP01", "--temperature", "0.5", "--json"],
    "auto":        ["--map", "MAP02", "--no-assists", "--steps", "1000",
                    "--iterations", "1"],
    "notes":       [],
    "bestiary":    [],
    "behavior":    ["--trends", "--n-runs", "5"],
    "hypothesize": [],
    "semantic":    ["stats"],
    "recall":      ["deaths", "near", "door"],
    "bc":          [],
    "suggest":     [],
    "lessons":     [],
    "experiment":  ["--list"],
    "curriculum":  [],
    "curriculum2": ["--stages", "mywh", "--steps", "1000"],
    "prune":       [],          # dry-run by default: reads dirs, deletes nothing
    "report":      ["--out", "/tmp/hellmind_dryrun_report.html", "--last", "5"],
    "vault":       [],          # idempotent note backfill (writes into the real vault)
    "progress":    [],
    "timeline":    [],
    "knowledge":   [],
    "rollback":    [],
    "learned":     [],
}


@pytest.fixture()
def parser():
    return doom_cli.build_parser()


@pytest.mark.parametrize("command", sorted(DRYRUN_ARGV))
def test_cli_command_dry_runs(parser, command):
    """Parse real argv for the command and invoke its handler with execution mocked."""
    args = parser.parse_args([command] + DRYRUN_ARGV[command])
    assert hasattr(args, "fn"), f"{command} has no handler bound"

    captured = {}

    def fake_run(cmd, env=None, title=None, **kw):
        captured["cmd"] = cmd
        assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd), \
            f"{command}: cmd must be a list of strings, got {cmd!r}"
        return 0

    fake_proc = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(doom_cli, "run", side_effect=fake_run), \
         patch("subprocess.run", return_value=fake_proc), \
         patch("subprocess.Popen", return_value=fake_proc):
        rc = args.fn(args)

    assert rc in (0, None) or isinstance(rc, int), f"{command} returned {rc!r}"


def test_every_dryrun_command_exists_in_parser(parser):
    """Guard the curated list against command renames."""
    sub = next(a for a in parser._actions
               if isinstance(a, doom_cli.argparse._SubParsersAction))
    available = set(sub.choices)
    missing = set(DRYRUN_ARGV) - available
    assert not missing, f"dry-run list references unknown commands: {missing}"


# Tracker aggregates that MUST reach every consumer through METRICS_JSON. A metric
# doesn't exist until it's in build_metrics — the tracker computed route_progress for a
# full evaluation while the loop scored None (the inline dict was never extended).
TRACKER_HEADLINE_KEYS = [
    "kills_per_episode", "shooting_accuracy", "exit_rate", "exit_progress",
    "route_progress", "route_progress_best", "death_route_dist",
    "weapon_switches_per_episode", "enemies_seen_per_episode",
    "hits_taken_per_episode", "aim_offset", "wasted_shot_rate", "kill_conversion",
    "revisit_rate", "combat_fraction", "combat_engagement", "combat_accuracy",
]


def test_episode_stats_ci_math():
    """The t-CI must be exact: for [1..10], mean=5.5, s=3.0277, t(9)=2.262 →
    half-width = 2.262·3.0277/√10 = 2.1659."""
    from instrumentation.stats_tracker import StatsTracker
    tr = StatsTracker(["ATTACK"])
    tr.episode_rewards = list(range(1, 11))
    tr.episode_lengths = [100] * 10
    s = tr.snapshot(0)
    st = s["episode_stats"]["episode_reward"]
    assert st["n"] == 10
    assert st["mean"] == pytest.approx(5.5)
    assert st["std"] == pytest.approx(3.02765, abs=1e-4)
    half = (st["ci95_hi"] - st["ci95_lo"]) / 2
    assert half == pytest.approx(2.262 * 3.02765 / 10 ** 0.5, abs=1e-3)
    # n=1: degenerate interval, no crash
    assert s["episode_stats"]["episode_length"]["std"] >= 0


def test_thesis_report_renders_from_synthetic_trail(tmp_path):
    """End-to-end: trail → HTML with methodology, trajectory chart, regime cut,
    formulas and limitations. Regime boundary must be detected from the map switch."""
    import json as _json
    from writer.thesis_report import render, _regime_boundaries
    rows = []
    for i in range(8):
        rows.append({"iter": i, "score": 0.1 * i, "kept": i % 2 == 0,
                     "env": {"MAPS": "MAP01" if i < 4 else "MAP02"},
                     "plateau_level": 1 if i == 6 else 0,
                     "metrics": {"route_progress": 0.05 * i, "exit_rate": 0.0,
                                 "kills_per_episode": 5.0, "death_rate": 0.2,
                                 "timeout_rate": 0.8, "explored_fraction": 0.1,
                                 "eval_meta": {"episodes": 10, "temperature": 0.5,
                                               "seed": 42, "map": "MAP02",
                                               "assists": {"auto_aim": False}},
                                 "episode_stats": {"episode_reward": {
                                     "mean": 1.0, "std": 0.5, "n": 10,
                                     "median": 1.0, "ci95_lo": 0.64,
                                     "ci95_hi": 1.36}}}})
    with open(tmp_path / "autonomy.jsonl", "w") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")
    cuts = _regime_boundaries(rows)
    assert 4 in cuts, "map switch at iter 4 must be a regime boundary"
    assert 6 in cuts, "plateau escape at iter 6 must be a regime boundary"
    html = render(str(tmp_path))
    for needle in ("Methodology", "1.000 ± 0.360", "Formulas", "Limitations",
                   "temperature=0.5", "data:image/png"):
        assert needle in html, f"report missing: {needle}"


def test_metrics_json_carries_every_headline_tracker_key():
    from rl.eval import build_metrics
    fake_summary = {
        "kills_per_episode": 1.0, "shooting_accuracy": 0.1, "success_rate": 0.5,
        "mean_base_reward": 0.0, "mean_episode_length": 100.0,
        "episodes": 10, "terminals": {"death": 5, "timeout": 5},
        "map_coverage": {"explored_fraction": 0.1, "cells_visited": 50},
    }
    metrics = build_metrics(fake_summary)
    missing = [k for k in TRACKER_HEADLINE_KEYS if k not in metrics]
    assert not missing, f"tracker metrics that never reach METRICS_JSON: {missing}"
