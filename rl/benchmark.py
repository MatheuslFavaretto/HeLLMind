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

METRIC_KEYS = ["exit_rate", "exit_progress", "explored_fraction", "kills_per_episode",
               "death_rate", "combat_engagement", "mean_base_reward"]


def _run_one(config: dict, seed: int, doom_map: str, steps: int, episodes: int,
             n_envs: int, ck_dir: str, algo: str = "ppo") -> dict:
    """Train one (config, seed) fresh, then eval it. Returns the parsed metrics dict.
    algo='ppo' uses rl.train (PPO); algo='dqn' uses rl.train_dqn (QR-DQN)."""
    env = {**os.environ, **_BASE_OFF, **config,
           "CAMPAIGN": "1", "MAPS": doom_map, "SEED": str(seed),
           "N_ENVS": str(n_envs), "CHECKPOINT_DIR": ck_dir, "DOCS_ENABLED": "0",
           "MEMORY_ENABLED": config.get("MEMORY_ENABLED", "0")}
    train_module = "rl.train_dqn" if algo == "dqn" else "rl.train"
    train_args = (["--fresh", "--maps", doom_map, "--n-envs", str(n_envs),
                   "--timesteps", str(steps)] if algo == "ppo"
                  else ["--fresh", "--map", doom_map, "--n-envs", str(n_envs),
                        "--timesteps", str(steps)])
    subprocess.run([PY, "-m", train_module, *train_args], cwd=ROOT, env=env, check=True)
    out = subprocess.run([PY, "-m", "rl.eval", "--episodes", str(episodes),
                          "--json", "--temperature", "0.5"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("METRICS_JSON"):
            return json.loads(line.split("METRICS_JSON", 1)[1])
    raise RuntimeError(f"eval produced no METRICS_JSON for {config} seed {seed}:\n{out.stdout[-500:]}")


def run(doom_map="MAP01", steps=50000, seeds=(42, 123), episodes=20, n_envs=4,
        configs=None, out_dir=None, algo="ppo") -> dict:
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
            per_seed.append(_run_one(CONFIGS[name], seed, doom_map, steps, episodes, n_envs, ck, algo))
            done += 1
            elapsed = time.time() - t0
            per_run = elapsed / done
            remaining = per_run * (total - done)
            eta = datetime.now().timestamp() + remaining
            print(f"[benchmark] ✓ {done}/{total} done · this run {_fmt(time.time() - r_start)} · "
                  f"elapsed {_fmt(elapsed)} · ~{_fmt(remaining)} left"
                  + (f" · ETA {datetime.fromtimestamp(eta).strftime('%H:%M')}" if remaining > 0
                     else " · DONE"))
        agg = _aggregate(per_seed)
        agg["_score"] = _config_score(agg)   # one rankable number per config
        results[name] = agg

    best = max(results, key=lambda n: results[n]["_score"]) if results else None
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "map": doom_map, "steps": steps, "seeds": list(seeds), "episodes": episodes,
        "best": best, "configs": results,
    }
    _write_json(out_dir, payload)
    _write_csv(out_dir, results)
    _write_md(out_dir, payload)
    _write_html(out_dir, payload)
    print(f"\n[benchmark] wrote results to {out_dir}/ "
          f"(benchmark.json/.csv/.md/.html) · best config: {best}")
    return payload


def _fmt(seconds: float) -> str:
    """Human-readable duration: '45s', '3m12s', '1h04m'."""
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _config_score(agg: dict) -> float:
    """One composite score per config so they can be ranked at a glance — same philosophy as
    the auto loop: finishing > approaching the exit > exploring > fighting, minus dying."""
    g = lambda k: float(agg.get(k, {}).get("mean", 0.0))
    kills = min(g("kills_per_episode"), 5.0) / 5.0
    return round(4.0 * g("exit_rate") + 1.5 * g("exit_progress")
                 + 3.0 * g("explored_fraction") + 0.5 * kills
                 - 1.0 * g("death_rate"), 3)


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


_PCT = {"exit_rate", "exit_progress", "explored_fraction", "combat_engagement", "death_rate"}


def _cell(k, agg):
    m, sd = agg[k]["mean"], agg[k]["std"]
    return f"{m*100:.0f}±{sd*100:.0f}%" if k in _PCT else f"{m:.2f}±{sd:.2f}"


def _write_md(out_dir, payload):
    results, best = payload["configs"], payload.get("best")
    lines = ["# 📊 HeLLMind ablation benchmark", "",
             f"_map {payload['map']} · {payload['steps']:,} steps · "
             f"{len(payload['seeds'])} seeds · eval {payload['episodes']} eps (tempered)_", "",
             "Does each layer add value? **score** ranks each config (finishing > approaching "
             "the exit > exploring > fighting − dying). Mean ± std across seeds.", "",
             "| config | score | " + " | ".join(METRIC_KEYS) + " |",
             "|" + "---|" * (len(METRIC_KEYS) + 2)]
    for name, agg in results.items():
        star = " ⭐" if name == best else ""
        cells = " | ".join(_cell(k, agg) for k in METRIC_KEYS)
        lines.append(f"| **{name}**{star} | **{agg.get('_score', 0):.2f}** | {cells} |")
    lines += ["", f"> Best: **{best}**. Reproduce: `doom-cli benchmark`. "
              "Open `benchmark.html` for the visual report."]
    with open(os.path.join(out_dir, "benchmark.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_html(out_dir, payload):
    """A self-contained dark/ember HTML report — score bars + the full metric table."""
    results, best = payload["configs"], payload.get("best")
    scores = {n: a.get("_score", 0.0) for n, a in results.items()}
    smax = max(scores.values()) or 1.0
    smin = min(scores.values())
    span = (smax - smin) or 1.0

    rows = ""
    for name, agg in results.items():
        is_best = name == best
        frac = (scores[name] - smin) / span
        bar_w = 8 + frac * 92  # 8–100% width so even the worst shows a sliver
        cells = "".join(f"<td>{_cell(k, agg)}</td>" for k in METRIC_KEYS)
        rows += (
            f'<tr class="{"best" if is_best else ""}">'
            f'<td class="cfg">{name}{" ⭐" if is_best else ""}</td>'
            f'<td class="score"><div class="bar" style="width:{bar_w:.0f}%">'
            f'{scores[name]:.2f}</div></td>{cells}</tr>\n')
    heads = "".join(f"<th>{k.replace('_', ' ')}</th>" for k in METRIC_KEYS)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>HeLLMind benchmark</title><style>
body{{background:#15110d;color:#f3e9dc;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:40px}}
h1{{color:#ff5a00;margin:0 0 4px}} .sub{{color:#9b8e7e;margin-bottom:24px}}
table{{border-collapse:collapse;width:100%;max-width:1100px}}
th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid #3a2c1e}}
th{{color:#ffd000;font-weight:600;text-align:right}} th:first-child,td.cfg{{text-align:left}}
td.cfg{{font-weight:700;color:#ffb060}} tr.best{{background:#241a10}}
.bar{{background:linear-gradient(90deg,#c41200,#ff9500);color:#15110d;font-weight:700;
padding:3px 8px;border-radius:4px;text-align:right;min-width:34px}}
.foot{{color:#9b8e7e;margin-top:20px;font-size:13px}}</style></head><body>
<h1>📊 HeLLMind ablation benchmark</h1>
<div class="sub">map {payload['map']} · {payload['steps']:,} steps · {len(payload['seeds'])} seeds ·
eval {payload['episodes']} eps (tempered) · best: <b>{best}</b></div>
<table><thead><tr><th>config</th><th>score</th>{heads}</tr></thead>
<tbody>{rows}</tbody></table>
<div class="foot">score = finishing &gt; approaching the exit &gt; exploring &gt; fighting − dying.
Generated {payload['generated']}.</div></body></html>"""
    with open(os.path.join(out_dir, "benchmark.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    p = argparse.ArgumentParser(description="Reproducible ablation benchmark.")
    p.add_argument("--map", default="MAP01")
    p.add_argument("--steps", type=int, default=50000, help="Train steps per config/seed.")
    p.add_argument("--seeds", default="42,123", help="Comma-separated seeds.")
    p.add_argument("--episodes", type=int, default=20, help="Eval episodes.")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--configs", default=None,
                   help="Comma-separated subset (default: all: baseline,rnd,memory,full).")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"],
                   help="Algorithm: ppo (default/PPO) or dqn (QR-DQN, V2 engine).")
    args = p.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    configs = args.configs.split(",") if args.configs else None
    run(doom_map=args.map, steps=args.steps, seeds=seeds, episodes=args.episodes,
        n_envs=args.n_envs, configs=configs, algo=args.algo)


if __name__ == "__main__":
    main()
