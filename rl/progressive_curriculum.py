"""Progressive curriculum engine (V2 Phase 2).

The V1 mistake: training directly on full maps (hard). The agent had to simultaneously
learn navigation, combat, and survival — with 80% death-rate as the result.

This curriculum trains one skill at a time using the SAME action space and brain,
so weights transfer between stages without any architecture mismatch:

  Stage 1 — NAVIGATE: find the exit. Max exploration/exit rewards, combat zeroed.
  Stage 2 — SURVIVE:  navigate without dying. Death penalty raised, combat added.
  Stage 3 — FULL:     all rewards live. The agent has the building blocks.

Transfer is trivial because the action space never changes. The same QR-DQN or PPO
brain is loaded at the start of each stage from the previous stage's checkpoint.

    python -m rl.progressive_curriculum --map MAP01 --steps-per-stage 150000
    python -m rl.progressive_curriculum --algo dqn --steps-per-stage 200000
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Reward profiles per stage (env-var overrides) ─────────────────────────────

STAGE_PROFILES = {
    "navigate": {
        # Goal: find and reach the exit. Combat zeroed so the agent ignores enemies
        # and focuses entirely on movement and map coverage.
        "COVERAGE_REWARD": "2.0",
        "FRONTIER_REWARD": "0.1",
        "EXIT_REWARD": "1000.0",
        "EXIT_PROX_SCALE": "0.5",
        "GOEXPLORE_GOAL_PROB": "0.5",
        "USE_RND": "1", "RND_SCALE": "0.5",
        "HIT_REWARD": "0", "MISS_PENALTY": "0",
        "KILL_REWARD": "0",
        "DEATH_PENALTY": "1.0",           # tiny — don't obsess, just don't get stuck
        "DAMAGE_TAKEN_PENALTY": "0.05",
        "ENGAGEMENT_REWARD": "0",
        "BESTIARY_REWARD": "0",
        "EPISODE_TIMEOUT": "3500",
        "ENT_COEF": "0.05",               # higher entropy: force exploration
    },
    "survive": {
        # Goal: navigate without dying. Now combat matters: dying is expensive,
        # so the agent must learn to engage enemies or avoid them.
        "COVERAGE_REWARD": "1.0",
        "FRONTIER_REWARD": "0.05",
        "EXIT_REWARD": "500.0",
        "EXIT_PROX_SCALE": "0.3",
        "GOEXPLORE_GOAL_PROB": "0.3",
        "USE_RND": "1", "RND_SCALE": "0.3",
        "HIT_REWARD": "2.0", "MISS_PENALTY": "0.05",
        "KILL_REWARD": "5.0",
        "DEATH_PENALTY": "15.0",          # raised sharply: dying is really bad
        "DAMAGE_TAKEN_PENALTY": "0.3",
        "ENGAGEMENT_REWARD": "0.02",
        "BESTIARY_REWARD": "1",
        "EPISODE_TIMEOUT": "2800",
        "ENT_COEF": "0.03",
    },
    "full": {
        # Stage 3: restore normal defaults (just unset the overrides)
        # Memory and docs enabled for the real run.
    },
}


# ── Core ───────────────────────────────────────────────────────────────────────

def _run_stage(stage: str, profile: dict, doom_map: str,
               steps: int, algo: str, fresh: bool) -> None:
    env = {**os.environ, "CAMPAIGN": "1", "MAPS": doom_map,
           "DOCS_ENABLED": "0",
           "MEMORY_ENABLED": "1" if stage == "full" else "0",
           **profile}

    if algo == "dqn":
        cmd = [sys.executable, "-m", "rl.train_dqn",
               "--map", doom_map, "--timesteps", str(steps), "--n-envs", "1"]
    else:
        cmd = [sys.executable, "-m", "rl.train",
               "--maps", doom_map, "--timesteps", str(steps)]

    if fresh:
        cmd.append("--fresh")

    print(f"\n{'═'*64}")
    print(f"  STAGE: {stage.upper()}  |  algo: {algo}  |  steps: {steps:,}")
    key = {k: v for k, v in profile.items()
           if any(x in k for x in ("EXIT_REWARD", "DEATH_PENALTY", "COVERAGE_REWARD"))}
    if key:
        print(f"  reward profile: {key}")
    print(f"{'═'*64}\n")
    subprocess.run(cmd, cwd=ROOT, env=env, check=False)


def _eval_stage(profile: dict, doom_map: str, episodes: int = 20) -> dict:
    env = {**os.environ, "CAMPAIGN": "1", "MAPS": doom_map,
           "DOCS_ENABLED": "0", "MEMORY_ENABLED": "0", **profile}
    out = subprocess.run(
        [sys.executable, "-m", "rl.eval",
         "--episodes", str(episodes), "--json", "--temperature", "0.5"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("METRICS_JSON"):
            return json.loads(line.split("METRICS_JSON", 1)[1])
    return {}


def run(doom_map: str = "MAP01", steps_per_stage: int = 150_000,
        algo: str = "ppo", stages: list | None = None,
        eval_episodes: int = 20) -> dict:
    """Run the full progressive curriculum and return per-stage metrics."""
    stages = stages or ["navigate", "survive", "full"]
    out_dir = os.path.join(ROOT, "reports",
                           f"curriculum-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for i, stage in enumerate(stages):
        profile = STAGE_PROFILES.get(stage, {})
        _run_stage(stage, profile, doom_map, steps_per_stage, algo, fresh=(i == 0))

        print(f"\n── eval: {stage} ──")
        m = _eval_stage(profile, doom_map, eval_episodes)
        results[stage] = m
        if m:
            print(f"  explored={m.get('explored_fraction',0):.0%}  "
                  f"exit={m.get('exit_rate',0):.0%}  "
                  f"→exit={m.get('exit_progress',0):.0%}  "
                  f"death={m.get('death_rate',0):.0%}  "
                  f"kills={m.get('kills_per_episode',0):.1f}")

        with open(os.path.join(out_dir, f"stage{i+1}_{stage}.json"), "w") as f:
            json.dump({"stage": stage, "metrics": m}, f, indent=2)

    _write_summary(out_dir, results, doom_map, algo)
    print(f"\n✅  curriculum done → {out_dir}/summary.md")
    return results


def _write_summary(out_dir, results, doom_map, algo) -> None:
    pct = {"exit_rate", "exit_progress", "explored_fraction", "death_rate"}
    keys = ["exit_rate", "exit_progress", "explored_fraction",
            "death_rate", "kills_per_episode"]
    lines = [f"# 📈 Curriculum — {doom_map} ({algo})", "",
             "| stage | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for stage, m in results.items():
        cells = " | ".join(
            f"{m.get(k,0)*100:.0f}%" if k in pct else f"{m.get(k,0):.2f}"
            for k in keys)
        lines.append(f"| **{stage}** | {cells} |")
    lines += ["",
              "> Same brain across all stages — weights transfer via checkpoint, no mismatch.",
              f"> Reproduce: `doom-cli curriculum2 --algo {algo}`"]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Progressive curriculum (V2 Phase 2).")
    p.add_argument("--map", default="MAP01")
    p.add_argument("--steps-per-stage", type=int, default=150_000)
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--stages", default="navigate,survive,full")
    p.add_argument("--eval-episodes", type=int, default=20)
    args = p.parse_args()
    run(doom_map=args.map, steps_per_stage=args.steps_per_stage,
        algo=args.algo, stages=args.stages.split(","),
        eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
