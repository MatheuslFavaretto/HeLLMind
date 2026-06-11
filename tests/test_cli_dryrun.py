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
