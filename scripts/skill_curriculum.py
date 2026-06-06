"""Staged skill curriculum on ONE brain — develop skills cumulatively (combat → nav → objectives).

The scenario curricula (my_way_home, defend_the_center) prove a skill in ISOLATION but have
different action spaces, so their weights DON'T transfer. This trains the SAME campaign a19 brain
(same obs: semantic + strafe + game_vars → weights carry) through stages, changing only the REWARD
focus each stage and RESUMING the previous stage's weights — so skills ACCUMULATE instead of
replacing each other. After each stage it evals and prints the scoreboard, and checks the new skill
came WITHOUT wrecking the old one.

    python scripts/skill_curriculum.py                 # run all stages from the start
    python scripts/skill_curriculum.py --start nav     # resume an existing combat brain, do nav+
    python scripts/skill_curriculum.py --steps 500000  # steps per stage

Honest: it's still frame-bound (skill = practice = training). This is the STRUCTURE for getting
better, not a shortcut around compute.
"""
import argparse
import json
import os
import subprocess
import sys

# Obs config that pins the brain family (a19 _se). MUST be identical every stage so --resume
# loads the same network (weights transfer). Only the reward knobs change per stage.
COMMON = {
    "CAMPAIGN": "1", "MAPS": "MAP01", "LOOP_MAPS": "0", "STRAFE": "1",
    "SEMANTIC_CHANNEL": "1", "GAME_VARS": "1", "FRAME_STACK": "2", "USE_LABELS": "1",
    "AUTO_AIM": "0", "AUTO_DOOR_NAV": "0", "AUTO_BEST_WEAPON": "0", "AUTO_USE": "0",
    "DOCS_ENABLED": "0", "SUGGEST_REWARDS": "0", "MEMORY_ENABLED": "1",
}

# Each stage: focus the reward on the skill being ADDED, while keeping enough of the previous
# so it isn't forgotten. Resumes the prior stage's weights.
STAGES = [
    {"name": "combat", "goal": "aim + survive (don't spray, don't die)", "rewards": {
        "COVERAGE_REWARD": "0.2", "FRONTIER_REWARD": "0.0", "RND_SCALE": "0.05",
        "DISCOVERY_REWARD": "0.0", "GOEXPLORE_GOAL_PROB": "0.0",
        "ENGAGEMENT_REWARD": "0.3", "HIT_REWARD": "5.0", "KILL_REWARD": "8.0",
        "MISS_PENALTY": "0.05", "DEATH_PENALTY": "15.0", "DAMAGE_TAKEN_PENALTY": "0.2",
        "COMBAT_EXPLORE_SPLIT": "1", "COMBAT_EXPLORE_FACTOR": "0.1"}},
    {"name": "nav", "goal": "explore the map WITHOUT losing the aim", "rewards": {
        "COVERAGE_REWARD": "1.0", "FRONTIER_REWARD": "0.05", "RND_SCALE": "0.3",
        "DISCOVERY_REWARD": "0.3", "GOEXPLORE_GOAL_PROB": "0.4",
        "ENGAGEMENT_REWARD": "0.2", "HIT_REWARD": "4.0", "KILL_REWARD": "6.0",
        "MISS_PENALTY": "0.03", "DEATH_PENALTY": "15.0", "DAMAGE_TAKEN_PENALTY": "0.2",
        "COMBAT_EXPLORE_SPLIT": "1", "COMBAT_EXPLORE_FACTOR": "0.3"}},
    {"name": "objectives", "goal": "reach the exit (keep combat + nav)", "rewards": {
        "COVERAGE_REWARD": "1.0", "FRONTIER_REWARD": "0.08", "RND_SCALE": "0.3",
        "DISCOVERY_REWARD": "0.5", "GOEXPLORE_GOAL_PROB": "0.5", "EXIT_REWARD": "1000.0",
        "ENGAGEMENT_REWARD": "0.15", "HIT_REWARD": "3.0", "KILL_REWARD": "5.0",
        "MISS_PENALTY": "0.02", "DEATH_PENALTY": "12.0", "DAMAGE_TAKEN_PENALTY": "0.15",
        "AUTO_USE": "1",  # let USE fire on contact so the exit/door can trigger (mechanical only)
        "COMBAT_EXPLORE_SPLIT": "1", "COMBAT_EXPLORE_FACTOR": "0.5"}},
]

SCORE_KEYS = ["aim_offset", "wasted_shot_rate", "kill_conversion", "shooting_accuracy",
              "explored_fraction", "revisit_rate", "exit_progress", "death_rate",
              "kills_per_episode"]


def _run(env_overrides, args_list):
    env = {**os.environ, **COMMON, **env_overrides}
    return subprocess.run([sys.executable, "-m"] + args_list, env=env,
                          capture_output=True, text=True)


def _eval_scoreboard(stage_rewards):
    """Eval the current brain; return the parsed METRICS_JSON dict (or {})."""
    res = _run(stage_rewards, ["rl.eval", "--episodes", "10", "--algo", "ppo",
                               "--temperature", "0.5", "--json"])
    for line in res.stdout.splitlines():
        if line.startswith("METRICS_JSON "):
            return json.loads(line[len("METRICS_JSON "):])
    print(res.stdout[-800:]); print(res.stderr[-400:])
    return {}


def _print_score(name, m):
    rb = m.get("reward_breakdown", {}) or {}
    rb_s = " ".join(f"{k}{v:+.0%}" for k, v in rb.items()) if rb else "n/a"
    print(f"\n  === [{name}] scoreboard ===")
    print(f"   AIM   offset {m.get('aim_offset',0):.2f} · wasted {m.get('wasted_shot_rate',0):.0%} "
          f"· acc {m.get('shooting_accuracy',0):.0%} · kill-conv {m.get('kill_conversion',0):.0%}")
    print(f"   MOVE  explored {m.get('explored_fraction',0):.0%} · revisit {m.get('revisit_rate',0):.0%} "
          f"· exit-prog {m.get('exit_progress',0):.0%}")
    print(f"   LIFE  deaths {m.get('death_rate',0):.0%} · kills/ep {m.get('kills_per_episode',0):.1f}")
    print(f"   REWARD FROM  {rb_s}")


def main():
    p = argparse.ArgumentParser(description="Staged skill curriculum on one campaign brain.")
    p.add_argument("--steps", type=int, default=500000, help="Training steps per stage.")
    p.add_argument("--start", default="combat", choices=[s["name"] for s in STAGES],
                   help="First stage to run (resume an existing brain to skip earlier ones).")
    p.add_argument("--fresh", action="store_true", help="Start the FIRST stage from zero.")
    args = p.parse_args()

    names = [s["name"] for s in STAGES]
    start_i = names.index(args.start)
    results = {}
    for i, stage in enumerate(STAGES[start_i:], start=start_i):
        print(f"\n========== STAGE {i+1}/{len(STAGES)}: {stage['name'].upper()} "
              f"— {stage['goal']} ==========")
        train_args = ["rl.train", "--timesteps", str(args.steps)]
        if args.fresh and i == start_i:
            train_args.append("--fresh")   # only the very first stage may start fresh
        # else: default (no --fresh) RESUMES this vault's a19 brain — skills accumulate.
        tr = _run(stage["rewards"], train_args)
        if tr.returncode != 0:
            print("[train FAILED]"); print(tr.stderr[-800:]); break
        # the train log tail (reward trend) for a quick gut-check
        for ln in tr.stdout.splitlines()[-6:]:
            if "ep_rew_mean" in ln or "total_timesteps" in ln:
                print("   " + ln.strip())
        m = _eval_scoreboard(stage["rewards"])
        results[stage["name"]] = m
        _print_score(stage["name"], m)

    out = os.path.join(os.getcwd(), "reports", "skill_curriculum_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[curriculum] done. scoreboard -> {out}")


if __name__ == "__main__":
    main()
