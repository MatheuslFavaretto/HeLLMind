"""Reproducible ablation benchmark — the scientific backbone of the project.

Question it answers: does each layer (RND, persistent memory, the full agent) actually add
value, or is it just machinery? It trains each config for the SAME budget across multiple
seeds, evaluates honestly (tempered T=0.5), and writes results/ (csv + json + md) with
mean ± std so wins aren't luck.

    python -m rl.benchmark                      # default: 4 configs x 2 seeds x 50k steps
    python -m rl.benchmark --steps 100000 --seeds 42,123,7 --map MAP01
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everything the ablation TOGGLES, pinned OFF so "baseline" is genuinely pure PPO. Perception
# channels are left at the env defaults (constant across configs) so the only difference is
# the layer under test.
_BASE_OFF = {
    "USE_RND": "0", "COVERAGE_REWARD": "0", "FRONTIER_REWARD": "0",
    "GOEXPLORE_GOAL_PROB": "0", "MEMORY_ENABLED": "0", "COMBAT_EXPLORE_SPLIT": "0",
    "BESTIARY_REWARD": "0", "DOCS_ENABLED": "0", "CONTROL_ENABLED": "0",
}
_EXPLORE = {"USE_RND": "1", "COVERAGE_REWARD": "0.5", "FRONTIER_REWARD": "0.05",
            "GOEXPLORE_GOAL_PROB": "0.4"}

# Cumulative layers: each adds one capability on top of the previous.
CONFIGS = {
    "baseline": {},                                              # pure PPO
    "rnd":      {**_EXPLORE},                                    # + curiosity/exploration
    "memory":   {**_EXPLORE, "MEMORY_ENABLED": "1", "BESTIARY_REWARD": "1"},  # + persistent memory
    "full":     {**_EXPLORE, "MEMORY_ENABLED": "1", "BESTIARY_REWARD": "1",
                 "COMBAT_EXPLORE_SPLIT": "1"},                   # + combat/explore decoupling
}

METRIC_KEYS = ["exit_rate", "explored_fraction", "kills_per_episode", "death_rate",
               "combat_engagement", "mean_base_reward"]


def _run_one(config: dict, seed: int, doom_map: str, steps: int, episodes: int,
             n_envs: int, ck_dir: str) -> dict:
    """Train one (config, seed) fresh, then eval it. Returns the parsed metrics dict."""
    env = {**os.environ, **_BASE_OFF, **config,
           "CAMPAIGN": "1", "MAPS": doom_map, "SEED": str(seed),
           "N_ENVS": str(n_envs), "CHECKPOINT_DIR": ck_dir, "DOCS_ENABLED": "0",
           "MEMORY_ENABLED": config.get("MEMORY_ENABLED", "0")}
    subprocess.run([PY, "-m", "rl.train", "--fresh", "--maps", doom_map,
                    "--n-envs", str(n_envs), "--timesteps", str(steps)],
                   cwd=ROOT, env=env, check=True)
    out = subprocess.run([PY, "-m", "rl.eval", "--episodes", str(episodes),
                          "--json", "--temperature", "0.5"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("METRICS_JSON"):
            return json.loads(line.split("METRICS_JSON", 1)[1])
    raise RuntimeError(f"eval produced no METRICS_JSON for {config} seed {seed}:\n{out.stdout[-500:]}")


def run(doom_map="MAP01", steps=50000, seeds=(42, 123), episodes=20, n_envs=4,
        configs=None, out_dir=None) -> dict:
    """Run the full matrix and write results/. Returns the aggregated results dict."""
    configs = configs or list(CONFIGS.keys())
    out_dir = out_dir or os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    ck_root = os.path.join(ROOT, ".cache", "benchmark")

    import time
    total = len(configs) * len(seeds)
    print(f"[benchmark] {len(configs)} configs × {len(seeds)} seeds = {total} runs "
          f"× {steps:,} steps on {doom_map}. Timing the first run to estimate the rest…")
    t0 = time.time()
    done = 0
    results = {}
    for name in configs:
        per_seed = []
        for seed in seeds:
            ck = os.path.join(ck_root, f"{name}_s{seed}")
            os.makedirs(ck, exist_ok=True)
            print(f"\n===== [{done + 1}/{total}] {name} | seed {seed} | {steps:,} steps =====")
            r_start = time.time()
            per_seed.append(_run_one(CONFIGS[name], seed, doom_map, steps, episodes, n_envs, ck))
            done += 1
            elapsed = time.time() - t0
            per_run = elapsed / done
            remaining = per_run * (total - done)
            eta = datetime.now().timestamp() + remaining
            print(f"[benchmark] ✓ {done}/{total} done · this run {_fmt(time.time() - r_start)} · "
                  f"elapsed {_fmt(elapsed)} · ~{_fmt(remaining)} left"
                  + (f" · ETA {datetime.fromtimestamp(eta).strftime('%H:%M')}" if remaining > 0
                     else " · DONE"))
        results[name] = _aggregate(per_seed)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "map": doom_map, "steps": steps, "seeds": list(seeds), "episodes": episodes,
        "configs": results,
    }
    _write_json(out_dir, payload)
    _write_csv(out_dir, results)
    _write_md(out_dir, payload)
    print(f"\n[benchmark] wrote results to {out_dir}/ (benchmark.json/.csv/.md)")
    return payload


def _fmt(seconds: float) -> str:
    """Human-readable duration: '45s', '3m12s', '1h04m'."""
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _aggregate(per_seed: list) -> dict:
    """mean ± std (across seeds) for each metric."""
    agg = {}
    for k in METRIC_KEYS:
        xs = [float(r.get(k, 0.0)) for r in per_seed]
        agg[k] = {"mean": statistics.fmean(xs) if xs else 0.0,
                  "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}
    agg["_seeds"] = len(per_seed)
    return agg


def _write_json(out_dir, payload):
    with open(os.path.join(out_dir, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(out_dir, results):
    with open(os.path.join(out_dir, "benchmark.csv"), "w", encoding="utf-8") as f:
        f.write("config," + ",".join(f"{k}_mean,{k}_std" for k in METRIC_KEYS) + "\n")
        for name, agg in results.items():
            row = [name] + [f"{agg[k]['mean']:.4f},{agg[k]['std']:.4f}" for k in METRIC_KEYS]
            f.write(",".join(row) + "\n")


def _write_md(out_dir, payload):
    results = payload["configs"]
    pct = {"exit_rate", "explored_fraction", "combat_engagement", "death_rate"}
    lines = ["# 📊 HeLLMind ablation benchmark", "",
             f"_map {payload['map']} · {payload['steps']:,} steps · "
             f"{len(payload['seeds'])} seeds · eval {payload['episodes']} eps (tempered)_", "",
             "Does each layer add value? Mean ± std across seeds.", "",
             "| config | " + " | ".join(METRIC_KEYS) + " |",
             "|" + "---|" * (len(METRIC_KEYS) + 1)]
    for name, agg in results.items():
        cells = []
        for k in METRIC_KEYS:
            m, sd = agg[k]["mean"], agg[k]["std"]
            cells.append(f"{m*100:.0f}±{sd*100:.0f}%" if k in pct else f"{m:.2f}±{sd:.2f}")
        lines.append(f"| **{name}** | " + " | ".join(cells) + " |")
    lines += ["", "> Reproduce: `doom-cli benchmark`. Raw numbers in `benchmark.json/.csv`."]
    with open(os.path.join(out_dir, "benchmark.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Reproducible ablation benchmark.")
    p.add_argument("--map", default="MAP01")
    p.add_argument("--steps", type=int, default=50000, help="Train steps per config/seed.")
    p.add_argument("--seeds", default="42,123", help="Comma-separated seeds.")
    p.add_argument("--episodes", type=int, default=20, help="Eval episodes.")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--configs", default=None,
                   help="Comma-separated subset (default: all: baseline,rnd,memory,full).")
    args = p.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    configs = args.configs.split(",") if args.configs else None
    run(doom_map=args.map, steps=args.steps, seeds=seeds, episodes=args.episodes,
        n_envs=args.n_envs, configs=configs)


if __name__ == "__main__":
    main()
