"""Experiment Engine (Phase 3) — validate hypotheses with honest multi-seed A/B.

Takes a hypothesis (from writer.hypothesize or the SQLite DB), derives a config delta,
runs a control vs experimental pair, judges the outcome on RAW metrics (not noisy
training reward), and records the verdict with confidence back to the DB + vault.

    python -m rl.experiment --hypothesis 1 --steps 200000 --seeds 42,123
    python -m rl.experiment --list    # show open hypotheses
    python -m rl.experiment --dry-run # show what would run, no training
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

PY = sys.executable

# Minimum improvement fraction to call a result "improved" vs "no_effect".
_IMPROVEMENT_MIN = 0.05   # 5% relative gain on the target metric
_REGRESSION_MIN  = 0.05   # 5% relative loss


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentPlan:
    hypothesis_id: int
    title: str
    metric: str
    direction: str          # 'up' or 'down'
    control_env: Dict       # base env vars (from .env)
    experimental_env: Dict  # control + config_delta
    seeds: List[int]
    steps: int


@dataclass
class ExperimentResult:
    plan: ExperimentPlan
    control_metrics: List[Dict]       # one dict per seed
    experimental_metrics: List[Dict]  # one dict per seed
    verdict: str                      # 'improved' | 'regressed' | 'no_effect'
    confidence: float
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _subprocess_env(extra: dict) -> dict:
    return {**os.environ, **{k: str(v) for k, v in extra.items()}}


def _run_eval(env: dict, episodes: int) -> dict:
    out = subprocess.run(
        [PY, "-m", "rl.eval", "--episodes", str(episodes), "--json"],
        env=_subprocess_env(env), check=True, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("METRICS_JSON "):
            return json.loads(line[len("METRICS_JSON "):])
    raise RuntimeError("eval produced no METRICS_JSON")


def _run_train(env: dict, doom_map: str, steps: int, fresh: bool) -> None:
    cmd = [PY, "-m", "rl.train", "--maps", doom_map,
           "--n-envs", env.get("N_ENVS", "4"), "--timesteps", str(steps)]
    cmd.append("--fresh" if fresh else "--resume")
    subprocess.run(cmd, env=_subprocess_env(env), check=True)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def plan_from_hypothesis(
    cfg,
    hypothesis_id: int,
    steps: int,
    seeds: List[int],
    doom_map: str = None,
) -> ExperimentPlan:
    """Build an ExperimentPlan from a hypothesis stored in the SQLite DB."""
    from writer.db import query_hypotheses

    rows = query_hypotheses(cfg.memory_dir, status="open")
    row = next((r for r in rows if r["id"] == hypothesis_id), None)
    if row is None:
        raise ValueError(f"Hypothesis {hypothesis_id} not found or not open.")

    # Parse the config_delta out of the hypothesis body (stored as JSON string in 'body').
    # The body is free text, but hypothesize.py encodes the delta via the template's
    # config_delta. We re-derive it by matching the title to a rule.
    from writer.hypothesize import _RULES
    template = next(
        (h for h in _RULES.values() if h.title == row["title"]), None
    )
    delta = template.config_delta if template else {}

    control_env = {
        "CAMPAIGN": "1", "MAPS": doom_map or cfg.maps[0],
        "DOCS_ENABLED": "0", "MEMORY_ENABLED": "0", "CONTROL_ENABLED": "0",
        "N_ENVS": str(cfg.n_envs),
        "COVERAGE_REWARD": str(cfg.coverage_reward),
        "EXIT_REWARD": str(cfg.exit_reward),
        "HIT_REWARD": str(cfg.hit_reward),
        "MISS_PENALTY": str(cfg.miss_penalty),
        "FRONTIER_REWARD": str(cfg.frontier_reward),
        "MOVE_REWARD": str(cfg.move_reward),
        "LIVING_REWARD": str(cfg.living_reward),
        "KILL_REWARD": str(cfg.kill_reward),
        "DEATH_PENALTY": str(cfg.death_penalty),
        "EPISODE_TIMEOUT": str(cfg.episode_timeout),
    }
    experimental_env = {**control_env, **delta}

    return ExperimentPlan(
        hypothesis_id=hypothesis_id,
        title=row["title"],
        metric=row["metric"],
        direction=row["direction"],
        control_env=control_env,
        experimental_env=experimental_env,
        seeds=seeds,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_experiment(
    plan: ExperimentPlan,
    eval_episodes: int = 15,
    verbose: bool = True,
) -> ExperimentResult:
    """Execute the A/B experiment: train+eval control vs experimental, multi-seed."""
    if verbose:
        print(f"\n[experiment] Testing H{plan.hypothesis_id}: {plan.title[:80]}")
        print(f"  metric={plan.metric}  direction={plan.direction}")
        print(f"  seeds={plan.seeds}  steps={plan.steps}")

    doom_map = plan.control_env.get("MAPS", "MAP01")
    control_metrics: List[Dict] = []
    experimental_metrics: List[Dict] = []

    for seed in plan.seeds:
        seed_str = str(seed)

        # Control
        ctrl_env = {**plan.control_env, "SEED": seed_str}
        if verbose:
            print(f"\n  [control  seed={seed}] training {plan.steps} steps...")
        _run_train(ctrl_env, doom_map, plan.steps, fresh=True)
        ctrl_m = _run_eval(ctrl_env, eval_episodes)
        control_metrics.append(ctrl_m)
        if verbose:
            print(f"  [control  seed={seed}] {plan.metric}="
                  f"{ctrl_m.get(plan.metric, '?'):.4f}")

        # Experimental
        exp_env = {**plan.experimental_env, "SEED": seed_str}
        if verbose:
            print(f"  [experimental seed={seed}] training {plan.steps} steps...")
        _run_train(exp_env, doom_map, plan.steps, fresh=True)
        exp_m = _run_eval(exp_env, eval_episodes)
        experimental_metrics.append(exp_m)
        if verbose:
            print(f"  [experimental seed={seed}] {plan.metric}="
                  f"{exp_m.get(plan.metric, '?'):.4f}")

    verdict, confidence, notes = _judge(plan, control_metrics, experimental_metrics)

    if verbose:
        print(f"\n  VERDICT: {verdict}  (confidence={confidence:.0%})")
        print(f"  {notes}")

    return ExperimentResult(
        plan=plan,
        control_metrics=control_metrics,
        experimental_metrics=experimental_metrics,
        verdict=verdict,
        confidence=confidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _judge(
    plan: ExperimentPlan,
    ctrl: List[Dict],
    exp: List[Dict],
) -> Tuple[str, float, str]:
    metric = plan.metric

    ctrl_vals = [float(m.get(metric, 0.0)) for m in ctrl]
    exp_vals  = [float(m.get(metric, 0.0)) for m in exp]
    ctrl_mean = _mean(ctrl_vals)
    exp_mean  = _mean(exp_vals)

    if ctrl_mean <= 1e-9:
        # Can't compute relative change if baseline is zero — use absolute
        delta = exp_mean - ctrl_mean
        rel = delta
    else:
        rel = (exp_mean - ctrl_mean) / ctrl_mean

    # Expected direction matters
    effective_gain = rel if plan.direction == "up" else -rel

    n = len(plan.seeds)
    # Confidence scales with number of seeds × magnitude of effect
    conf = min(1.0, abs(effective_gain) * n / 2.0)

    if effective_gain >= _IMPROVEMENT_MIN:
        verdict = "improved"
    elif effective_gain <= -_REGRESSION_MIN:
        verdict = "regressed"
    else:
        verdict = "no_effect"

    notes = (
        f"control {metric}={ctrl_mean:.4f}  experimental={exp_mean:.4f}  "
        f"relative_change={rel:+.1%}  n_seeds={n}"
    )
    return verdict, round(conf, 2), notes


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def record_result(cfg, result: ExperimentResult) -> None:
    """Persist the result to the SQLite DB and update the hypothesis status."""
    from writer import db as _db

    # Determine new hypothesis status
    status = "confirmed" if result.verdict == "improved" else (
        "rejected" if result.verdict == "regressed" else "open"
    )
    _db.update_hypothesis_status(cfg.memory_dir, result.plan.hypothesis_id, status)

    delta_str = json.dumps({
        k: v for k, v in result.plan.experimental_env.items()
        if result.plan.control_env.get(k) != v
    })
    _db.insert_experiment(
        cfg.memory_dir,
        param=delta_str,
        old_val=str(result.plan.control_env.get(result.plan.metric, "")),
        new_val=str(result.plan.experimental_env.get(result.plan.metric, "")),
        result=result.verdict,
        confidence=result.confidence,
        hypothesis_id=result.plan.hypothesis_id,
        notes=result.notes,
    )


def write_experiment_note(cfg, result: ExperimentResult) -> str:
    """Write `70-hypotheses/Experiment-H<id>.md` to the vault. Returns path."""
    out_dir = os.path.join(cfg.vault_path, "70-hypotheses")
    os.makedirs(out_dir, exist_ok=True)
    hid = result.plan.hypothesis_id
    path = os.path.join(out_dir, f"Experiment-H{hid}.md")

    icon = {"improved": "✅", "regressed": "❌", "no_effect": "⚠️"}.get(result.verdict, "?")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        "---",
        "type: experiment",
        f"created: {ts}",
        f"hypothesis_id: {hid}",
        f"verdict: {result.verdict}",
        f"confidence: {result.confidence}",
        "tags: [experiment, cognitive-loop, doom-rl]",
        "---",
        "",
        f"# {icon} Experiment H{hid}: {result.plan.title}",
        "",
        f"**Verdict:** {result.verdict}  **Confidence:** {result.confidence:.0%}",
        "",
        f"**Metric:** `{result.plan.metric}`  **Direction:** {result.plan.direction}",
        "",
        f"**Notes:** {result.notes}",
        "",
        "## Config delta (control → experimental)",
        "```",
    ]
    for k, v in result.plan.experimental_env.items():
        ctrl_v = result.plan.control_env.get(k)
        if ctrl_v != v:
            lines.append(f"{k}: {ctrl_v} → {v}")
    lines += [
        "```", "",
        "## Per-seed results",
        "",
        f"| Seed | Control `{result.plan.metric}` | Experimental `{result.plan.metric}` |",
        "|------|------|------|",
    ]
    for i, seed in enumerate(result.plan.seeds):
        ctrl_v = result.control_metrics[i].get(result.plan.metric, "?") if i < len(result.control_metrics) else "?"
        exp_v  = result.experimental_metrics[i].get(result.plan.metric, "?") if i < len(result.experimental_metrics) else "?"
        lines.append(f"| {seed} | {ctrl_v} | {exp_v} |")
    lines += [
        "",
        f"## Link",
        f"← [[Hypotheses]] (H{hid})",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run a hypothesis-driven A/B experiment.")
    p.add_argument("--hypothesis", type=int, default=None,
                   help="Hypothesis ID to test (from `doom-cli hypotheses` or SQLite).")
    p.add_argument("--steps", type=int, default=200000,
                   help="Training steps per arm per seed.")
    p.add_argument("--seeds", default="42,123",
                   help="Comma-separated seeds (e.g. 42,123,777).")
    p.add_argument("--episodes", type=int, default=15,
                   help="Eval episodes per arm per seed.")
    p.add_argument("--map", default=None,
                   help="Map to train on (default: cfg.maps[0]).")
    p.add_argument("--list", action="store_true",
                   help="List open hypotheses and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would run without training.")
    args = p.parse_args()

    from config import Config
    cfg = Config()

    if args.list:
        from writer.db import query_hypotheses
        rows = query_hypotheses(cfg.memory_dir, status="open")
        if not rows:
            print("[experiment] No open hypotheses. Run `doom-cli hypothesize` first.")
        for r in rows:
            print(f"  H{r['id']} [{r['confidence']:.0%}] {r['title']}")
        return

    if args.hypothesis is None:
        p.error("--hypothesis <id> is required (use --list to see open hypotheses).")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    plan = plan_from_hypothesis(
        cfg, args.hypothesis, steps=args.steps, seeds=seeds, doom_map=args.map
    )

    if args.dry_run:
        print("[experiment] DRY RUN — would test:")
        print(f"  Hypothesis H{plan.hypothesis_id}: {plan.title}")
        print(f"  Seeds: {plan.seeds}  Steps: {plan.steps}")
        print(f"  Control env delta (vs current .env):")
        for k, v in plan.experimental_env.items():
            if plan.control_env.get(k) != v:
                print(f"    {k}: {plan.control_env.get(k)} → {v}")
        return

    result = run_experiment(plan, eval_episodes=args.episodes)
    record_result(cfg, result)
    path = write_experiment_note(cfg, result)
    print(f"\n[experiment] wrote {path}")
    print(f"[experiment] verdict={result.verdict}  confidence={result.confidence:.0%}")


if __name__ == "__main__":
    main()
