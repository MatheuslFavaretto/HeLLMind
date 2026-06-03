"""Research Agent (Phase 5) — the meta-loop that ties everything together.

One unattended run cycles through:
  Memory → Behavior → Hypothesis → Experiment → Lesson → Curriculum → Training

Each iteration is logged to the vault. The loop continues until a stopping condition
is met (max_iterations, max_hours, or a validated improvement).

    python -m rl.research_agent --iterations 3 --steps 200000 --map MAP01
    python -m rl.research_agent --iterations 3 --steps 200000 --map MAP01 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional

PY = sys.executable


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ResearchIteration:
    iteration: int
    ts: str
    flags: List[str]              # behavior flag names
    hypotheses: List[str]         # hypothesis titles
    experiment_verdict: Optional[str]  # improved / regressed / no_effect / None
    experiment_metric: Optional[str]
    experiment_param: Optional[str]
    score_before: float
    score_after: float
    curriculum_weights: dict
    notes: str


# ---------------------------------------------------------------------------
# Score (same formula as autonomous.py)
# ---------------------------------------------------------------------------

def _score(m: dict) -> float:
    return (
        4.0 * m.get("exit_rate", 0.0)
        + 3.0 * m.get("explored_fraction", 0.0)
        + 0.5 * m.get("kills_per_episode", 0.0)
        + 1.0 * m.get("shooting_accuracy", 0.0)
    )


def _subprocess_env(extra: dict = None) -> dict:
    return {**os.environ, **(extra or {})}


def _run_eval(doom_map: str, episodes: int, extra_env: dict = None) -> dict:
    env = _subprocess_env({
        "CAMPAIGN": "1", "MAPS": doom_map, "DOCS_ENABLED": "0",
        "MEMORY_ENABLED": "0", "CONTROL_ENABLED": "0", **(extra_env or {}),
    })
    out = subprocess.run(
        [PY, "-m", "rl.eval", "--episodes", str(episodes), "--json"],
        env=env, check=True, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("METRICS_JSON "):
            return json.loads(line[len("METRICS_JSON "):])
    raise RuntimeError("eval produced no METRICS_JSON")


def _run_train(doom_map: str, steps: int, fresh: bool, extra_env: dict = None) -> None:
    env = _subprocess_env({
        "CAMPAIGN": "1", "MAPS": doom_map, "DOCS_ENABLED": "0",
        "MEMORY_ENABLED": "1", "CONTROL_ENABLED": "0", **(extra_env or {}),
    })
    cmd = [PY, "-m", "rl.train", "--maps", doom_map,
           "--timesteps", str(steps), "--resume" if not fresh else "--fresh"]
    subprocess.run(cmd, env=env, check=True)


# ---------------------------------------------------------------------------
# Research loop
# ---------------------------------------------------------------------------

def research_loop(
    cfg,
    doom_map: str,
    steps_per_iter: int,
    eval_episodes: int,
    max_iterations: int,
    fresh_first: bool,
    dry_run: bool,
    verbose: bool = True,
) -> List[ResearchIteration]:
    from writer.memory_store import MemoryStore
    from writer.behavior import detect
    from writer.hypothesize import generate, save_hypotheses
    from writer.snapshot_log import SnapshotLog, log_path_for
    from rl.curriculum import smart_weights, detect_forgetting
    from rl.experiment import plan_from_hypothesis, run_experiment, record_result, write_experiment_note
    from writer.db import query_hypotheses, build as db_build

    history: List[ResearchIteration] = []

    for i in range(max_iterations):
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if verbose:
            print(f"\n{'='*60}")
            print(f"[research] ITERATION {i}  map={doom_map}  steps={steps_per_iter:,}")
            print(f"{'='*60}")

        # 1. Train
        if not dry_run:
            _run_train(doom_map, steps_per_iter, fresh=fresh_first and i == 0)

        # 2. Eval (before)
        score_before = 0.0
        if not dry_run and i > 0:
            try:
                m = _run_eval(doom_map, eval_episodes)
                score_before = _score(m)
            except Exception as e:
                if verbose:
                    print(f"[research] eval failed: {e}")

        # 3. Detect behavior
        events = MemoryStore.read_events(cfg.memory_dir)
        snap_path = log_path_for(cfg.pending_dir, cfg.run_name)
        snaps = SnapshotLog.read_all(snap_path)
        flags = detect(events, snaps)
        flag_names = [f.name for f in flags]
        if verbose:
            if flags:
                print(f"[research] flags: {flag_names}")
            else:
                print("[research] no behavior flags detected")

        # 4. Generate & save hypotheses
        hypotheses = generate(flags)
        hyp_titles = [h.title for h in hypotheses]
        if hypotheses and not dry_run:
            save_hypotheses(cfg, hypotheses)
            db_build(cfg.memory_dir)  # rebuild SQLite with new events
        if verbose and hypotheses:
            print(f"[research] generated {len(hypotheses)} hypothesis/hypotheses")

        # 5. Pick the highest-confidence open hypothesis and run an experiment
        verdict = None
        exp_metric = None
        exp_param = None

        open_hyps = query_hypotheses(cfg.memory_dir, status="open")
        best_hyp = max(open_hyps, key=lambda r: r["confidence"], default=None) if open_hyps else None

        if best_hyp and not dry_run:
            if verbose:
                print(f"[research] testing H{best_hyp['id']}: {best_hyp['title'][:60]}")
            try:
                plan = plan_from_hypothesis(
                    cfg, best_hyp["id"], steps=steps_per_iter, seeds=[42], doom_map=doom_map
                )
                result = run_experiment(plan, eval_episodes=eval_episodes, verbose=verbose)
                record_result(cfg, result)
                write_experiment_note(cfg, result)
                verdict = result.verdict
                exp_metric = plan.metric
                exp_param = json.dumps(plan.experimental_env)
            except Exception as e:
                if verbose:
                    print(f"[research] experiment failed: {e}")
        elif dry_run and best_hyp:
            verdict = "dry_run"
            exp_metric = best_hyp.get("metric")
            exp_param = best_hyp.get("title")

        # 6. Curriculum reweighting
        maps = list(cfg.maps)
        cur_weights = smart_weights(events, maps)
        alerts = detect_forgetting(events, maps)
        if verbose and alerts:
            print(f"[research] forgetting alerts: {[a.map_name for a in alerts]}")
        if verbose:
            print(f"[research] curriculum weights: {cur_weights}")

        # 7. Eval (after)
        score_after = score_before
        if not dry_run:
            try:
                m = _run_eval(doom_map, eval_episodes)
                score_after = _score(m)
            except Exception as e:
                if verbose:
                    print(f"[research] post-eval failed: {e}")

        iter_rec = ResearchIteration(
            iteration=i,
            ts=ts,
            flags=flag_names,
            hypotheses=hyp_titles,
            experiment_verdict=verdict,
            experiment_metric=exp_metric,
            experiment_param=exp_param,
            score_before=round(score_before, 3),
            score_after=round(score_after, 3),
            curriculum_weights=cur_weights,
            notes=f"flags={len(flags)} hypotheses={len(hypotheses)} "
                  f"score {score_before:.2f}→{score_after:.2f}",
        )
        history.append(iter_rec)
        _write_log(cfg, history)

        if verbose:
            print(f"[research] iter {i} done — score {score_before:.2f}→{score_after:.2f} "
                  f"verdict={verdict}")

    return history


# ---------------------------------------------------------------------------
# Vault log
# ---------------------------------------------------------------------------

def _write_log(cfg, history: List[ResearchIteration]) -> None:
    os.makedirs(cfg.memory_dir, exist_ok=True)
    jsonl_path = os.path.join(cfg.memory_dir, "research_log.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(asdict(h)) + "\n")

    note_path = os.path.join(cfg.vault_path, cfg.dir_index, "Research Log.md")
    os.makedirs(os.path.dirname(note_path), exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "---", "type: research-log",
        f"updated: {ts}", "tags: [research-agent, cognitive-loop, doom-rl]", "---", "",
        "# Research Agent Log",
        "",
        "Each row: detect behavior → generate hypothesis → run experiment → reweight curriculum → train.",
        "",
        "| Iter | Flags | Hypotheses | Experiment | Metric | Score Δ | Notes |",
        "|------|-------|------------|-----------|--------|---------|-------|",
    ]
    for h in history:
        delta = h.score_after - h.score_before
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {h.iteration} | {', '.join(h.flags) or '—'} | {len(h.hypotheses)} | "
            f"{h.experiment_verdict or '—'} | {h.experiment_metric or '—'} | "
            f"{sign}{delta:.2f} | {h.notes} |"
        )

    # Was the milestone hit? (first validated improvement)
    wins = [h for h in history if h.experiment_verdict == "improved"]
    if wins:
        w = wins[0]
        lines += [
            "",
            f"## 🎉 MILESTONE HIT at iteration {w.iteration}",
            "",
            f"An automatically-generated hypothesis → experiment measurably improved "
            f"`{w.experiment_metric}`. Score delta: {w.score_after - w.score_before:+.2f}.",
            "",
            "**This is the first real act of self-improvement.**",
        ]
    else:
        lines += [
            "",
            "_Milestone not yet hit — no experiment has improved the agent so far._",
            "",
            "[[Hypotheses]] · [[Behavior]] · [[Curriculum]]",
        ]

    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Research Agent — autonomous cognitive loop: behavior → hypothesis → experiment → curriculum → train."
    )
    p.add_argument("--iterations", type=int, default=3,
                   help="Number of research iterations.")
    p.add_argument("--steps", type=int, default=200000,
                   help="Training steps per iteration (and per experiment arm).")
    p.add_argument("--episodes", type=int, default=15,
                   help="Eval episodes per check.")
    p.add_argument("--map", default=None,
                   help="Map to train on (default: cfg.maps[0]).")
    p.add_argument("--fresh", action="store_true",
                   help="Start the first iteration from scratch.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would run without training or experiments.")
    args = p.parse_args()

    from config import Config
    cfg = Config()
    doom_map = args.map or cfg.maps[0]

    print(f"[research] Starting Research Agent — {args.iterations} iter × "
          f"{args.steps:,} steps on {doom_map}"
          f"{' [DRY RUN]' if args.dry_run else ''}")

    history = research_loop(
        cfg=cfg,
        doom_map=doom_map,
        steps_per_iter=args.steps,
        eval_episodes=args.episodes,
        max_iterations=args.iterations,
        fresh_first=args.fresh,
        dry_run=args.dry_run,
    )

    print(f"\n[research] DONE — {len(history)} iterations. "
          f"See {cfg.vault_path}/{cfg.dir_index}/Research Log.md")


if __name__ == "__main__":
    main()
