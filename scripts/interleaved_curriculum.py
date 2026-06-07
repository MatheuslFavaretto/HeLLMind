"""Interleaved two-skill curriculum on ONE brain: alternate AIM (MAP02, combat-dense) and
EXPLORATION (MAP01) in short blocks, resuming the same a19 campaign brain each time, so the
agent develops BOTH skills WITHOUT catastrophic forgetting (sequential train-aim-then-explore
would overwrite the aim; interleaving keeps both alive).

Both maps share the campaign action space → weights transfer. After each round it evals on BOTH
maps to show retention: aim measured on MAP02 (kills/accuracy), exploration on MAP01 (explored).

    python scripts/interleaved_curriculum.py --rounds 3 --block-steps 100000
"""
import argparse
import json
import os
import subprocess
import sys

# Same obs/action config every block → --resume loads the same brain (skills accumulate).
COMMON = {
    "CAMPAIGN": "1", "STRAFE": "1", "SEMANTIC_CHANNEL": "1", "GAME_VARS": "1",
    "FRAME_STACK": "2", "USE_LABELS": "1",
    "AUTO_AIM": "0", "AUTO_DOOR_NAV": "0", "AUTO_BEST_WEAPON": "0", "AUTO_USE": "0",
    "DOCS_ENABLED": "0", "SUGGEST_REWARDS": "0", "MEMORY_ENABLED": "1",
}
AIM = {"MAPS": "MAP02", "ENGAGEMENT_REWARD": "0.3", "HIT_REWARD": "5", "KILL_REWARD": "8",
       "MISS_PENALTY": "0.05", "DEATH_PENALTY": "15", "DAMAGE_TAKEN_PENALTY": "0.2",
       "COVERAGE_REWARD": "0.2", "FRONTIER_REWARD": "0", "RND_SCALE": "0.05",
       "GOEXPLORE_GOAL_PROB": "0", "COMBAT_EXPLORE_SPLIT": "1", "COMBAT_EXPLORE_FACTOR": "0.1"}
EXPLORE = {"MAPS": "MAP01", "COVERAGE_REWARD": "2.0", "FRONTIER_REWARD": "0.1", "RND_SCALE": "0.5",
           "GOEXPLORE_GOAL_PROB": "0.6", "DISCOVERY_REWARD": "0.5", "ENGAGEMENT_REWARD": "0.1",
           "HIT_REWARD": "3", "KILL_REWARD": "5", "DEATH_PENALTY": "12",
           "COMBAT_EXPLORE_SPLIT": "1", "COMBAT_EXPLORE_FACTOR": "0.3"}


def _run(env_extra, args_list):
    env = {**os.environ, **COMMON, **env_extra}
    return subprocess.run([sys.executable, "-m"] + args_list, env=env,
                          capture_output=True, text=True)


def _train(rewards, steps, fresh):
    a = ["rl.train", "--timesteps", str(steps)]
    if fresh:
        a.append("--fresh")
    r = _run(rewards, a)
    for ln in r.stdout.splitlines()[-5:]:
        if "ep_rew_mean" in ln or "total_timesteps" in ln:
            print("   " + ln.strip())
    if r.returncode != 0:
        print("[train FAILED]"); print(r.stderr[-600:])
    return r.returncode == 0


def _eval(rewards, label):
    r = _run(rewards, ["rl.eval", "--episodes", "5", "--algo", "ppo",
                       "--temperature", "0.5", "--json"])
    for line in r.stdout.splitlines():
        if line.startswith("METRICS_JSON "):
            m = json.loads(line[len("METRICS_JSON "):])
            print(f"   [{label}] kills/ep {m.get('kills_per_episode',0):.1f} · "
                  f"acc {m.get('shooting_accuracy',0):.0%} · wasted {m.get('wasted_shot_rate',0):.0%} · "
                  f"explored {m.get('explored_fraction',0):.0%}")
            return m
    return {}


def main():
    p = argparse.ArgumentParser(description="Interleaved aim(MAP02) ↔ exploration(MAP01) curriculum.")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--block-steps", type=int, default=100000)
    args = p.parse_args()

    results = []
    for r in range(args.rounds):
        print(f"\n========== ROUND {r+1}/{args.rounds} ==========")
        print(f"-- AIM block (MAP02, combat) --")
        if not _train(AIM, args.block_steps, fresh=(r == 0)):
            break
        print(f"-- EXPLORE block (MAP01) --")
        if not _train(EXPLORE, args.block_steps, fresh=False):
            break
        print(f"-- retention check (both maps) --")
        results.append({"round": r + 1,
                        "aim_MAP02": _eval(AIM, "AIM MAP02"),
                        "explore_MAP01": _eval(EXPLORE, "EXPLORE MAP01")})

    out = os.path.join(os.getcwd(), "reports", "interleaved_curriculum.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[curriculum] done. Did it keep BOTH skills across rounds? -> {out}")
    # write an HTML report of the last round's exploration brain state
    try:
        from writer.html_report import write_report
        if results:
            write_report(results[-1]["explore_MAP01"], "reports/interleaved_report.html",
                         meta={"map": "MAP01+MAP02 interleaved"})
            print("[curriculum] HTML -> reports/interleaved_report.html")
    except Exception as e:
        print(f"[curriculum] HTML skipped: {e}")


if __name__ == "__main__":
    main()
