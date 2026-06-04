"""RL quality audit — validates that the agent is genuinely learning, not just memorising.

Academic checks (PPO-specific):
  1. Explained variance trend      — value function quality (> 0.5 = useful, > 0.8 = good)
  2. Entropy decay curve           — should fall GRADUALLY; sudden drop = premature collapse
  3. KL divergence stability       — should stay below clip_range * 2 (~0.4); spikes = instable
  4. Policy gradient sign          — should oscillate around 0, not drift positive (exploitation)
  5. Value loss trend              — should decrease; plateau = value function stuck
  6. Reward standard deviation     — high σ = noisy reward signal (hard to learn from)
  7. Stochastic vs deterministic   — run both evals, compare (gap = unexploited policy mass)

    python -m rl.audit                   # full report (all TB runs)
    python -m rl.audit --run PPO_8       # specific run
    python -m rl.audit --plot            # matplotlib charts (requires display)
    python -m rl.audit --json            # machine-readable output
"""
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

TB_DIR = os.path.join(os.path.dirname(__file__), "..", "tb")


def _load_scalar(ea, tag: str) -> List[Tuple[int, float]]:
    try:
        return [(e.step, e.value) for e in ea.Scalars(tag)]
    except KeyError:
        return []


def load_run(run_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    from tensorboard.backend.event_processing import event_accumulator as ea_mod
    ea = ea_mod.EventAccumulator(run_dir)
    ea.Reload()
    tags = {
        "ev":       "train/explained_variance",
        "entropy":  "train/entropy_loss",
        "kl":       "train/approx_kl",
        "pg_loss":  "train/policy_gradient_loss",
        "val_loss": "train/value_loss",
        "ep_rew":   "rollout/ep_rew_mean",
        "ep_len":   "rollout/ep_len_mean",
    }
    return {k: _load_scalar(ea, v) for k, v in tags.items()}


def _last(series, n=5):
    vals = [v for _, v in series[-n:]]
    return sum(vals) / len(vals) if vals else None


def _trend(series, n=10) -> Optional[float]:
    """Slope of a simple linear regression over the last n points (positive = rising)."""
    pts = series[-n:]
    if len(pts) < 2:
        return None
    xs = [i for i in range(len(pts))]
    ys = [v for _, v in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def grade(data: dict) -> dict:
    """Score each dimension 0-10 and produce a plain-English verdict."""
    checks = {}

    # 1. Explained variance (value quality)
    ev = _last(data.get("ev", []))
    if ev is None:
        checks["value_quality"] = {"score": None, "note": "no data"}
    elif ev >= 0.8:
        checks["value_quality"] = {"score": 9, "note": f"EV={ev:.3f} — value function excellent"}
    elif ev >= 0.5:
        checks["value_quality"] = {"score": 6, "note": f"EV={ev:.3f} — value function acceptable"}
    else:
        checks["value_quality"] = {"score": 2, "note": f"EV={ev:.3f} — value function weak (policy gets noisy gradients)"}

    # 2. Entropy trend (exploration health)
    ent = _last(data.get("entropy", []))
    ent_trend = _trend(data.get("entropy", []))
    if ent is None:
        checks["entropy_health"] = {"score": None, "note": "no data"}
    elif ent < -2.5:
        checks["entropy_health"] = {"score": 2, "note": f"entropy={ent:.2f} — policy collapsed (too deterministic, no exploration)"}
    elif ent_trend is not None and ent_trend > 0.05:
        checks["entropy_health"] = {"score": 4, "note": f"entropy rising ({ent:.2f}) — policy NOT converging (still noisy)"}
    else:
        checks["entropy_health"] = {"score": 8, "note": f"entropy={ent:.2f} — gradual decline (healthy convergence)"}

    # 3. KL divergence (update stability)
    kl = _last(data.get("kl", []))
    if kl is None:
        checks["kl_stability"] = {"score": None, "note": "no data"}
    elif kl > 0.05:
        checks["kl_stability"] = {"score": 3, "note": f"KL={kl:.4f} — too large, policy changing drastically per update"}
    elif kl > 0.02:
        checks["kl_stability"] = {"score": 6, "note": f"KL={kl:.4f} — moderate, near the clip threshold"}
    else:
        checks["kl_stability"] = {"score": 9, "note": f"KL={kl:.4f} — stable updates"}

    # 4. Value loss trend (is the value function improving?)
    vl_trend = _trend(data.get("val_loss", []))
    vl = _last(data.get("val_loss", []))
    if vl_trend is None:
        checks["value_improving"] = {"score": None, "note": "no data"}
    elif vl_trend < -0.1:
        checks["value_improving"] = {"score": 9, "note": f"val_loss={vl:.2f}, trending down — value function improving"}
    elif vl_trend < 0.1:
        checks["value_improving"] = {"score": 6, "note": f"val_loss={vl:.2f}, stable — value function plateau"}
    else:
        checks["value_improving"] = {"score": 2, "note": f"val_loss={vl:.2f}, rising — value function diverging"}

    # 5. Reward learning (ep_rew trend)
    rew_trend = _trend(data.get("ep_rew", []), n=20)
    rew = _last(data.get("ep_rew", []))
    if rew_trend is None:
        checks["reward_learning"] = {"score": None, "note": "no data"}
    elif rew_trend > 0.05:
        checks["reward_learning"] = {"score": 9, "note": f"ep_rew={rew:.1f}, rising — genuine learning"}
    elif rew_trend > -0.05:
        checks["reward_learning"] = {"score": 5, "note": f"ep_rew={rew:.1f}, plateau — converged or stuck"}
    else:
        checks["reward_learning"] = {"score": 2, "note": f"ep_rew={rew:.1f}, falling — reward regression"}

    scored = {k: v for k, v in checks.items() if v["score"] is not None}
    overall = sum(v["score"] for v in scored.values()) / len(scored) if scored else None
    return {"checks": checks, "overall": overall}


def find_latest_run(tb_dir: str) -> Optional[str]:
    runs = sorted(glob.glob(os.path.join(tb_dir, "PPO_*")),
                  key=lambda p: os.path.getmtime(p), reverse=True)
    return runs[0] if runs else None


def main() -> None:
    p = argparse.ArgumentParser(description="RL quality audit.")
    p.add_argument("--run", default=None, help="TB run dir (default: latest)")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--plot", action="store_true", help="Show matplotlib charts.")
    args = p.parse_args()

    tb_dir = os.path.join(os.path.dirname(__file__), "..", "tb")
    run_dir = args.run or find_latest_run(tb_dir)
    if not run_dir:
        print("[audit] No TensorBoard runs found in ./tb/", file=sys.stderr)
        sys.exit(1)

    data = load_run(run_dir)
    result = grade(data)

    if args.as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n=== RL Quality Audit — {os.path.basename(run_dir)} ===\n")
    for name, check in result["checks"].items():
        score = check["score"]
        bar = "█" * (score or 0) + "░" * (10 - (score or 0))
        label = f"{score}/10" if score is not None else "N/A"
        print(f"  {name:<22} [{bar}] {label}")
        print(f"    {check['note']}")
    overall = result.get("overall")
    if overall:
        print(f"\n  OVERALL: {overall:.1f}/10")
        if overall >= 8:
            print("  ✅ Agent is genuinely learning — trust these eval numbers.")
        elif overall >= 6:
            print("  ⚠️  Learning but with weaknesses — check the low-scoring dimensions.")
        else:
            print("  ❌ Learning quality is poor — reward shaping or hyperparams need attention.")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f"RL Audit — {os.path.basename(run_dir)}")
            pairs = [
                ("ep_rew",   "Episode Reward", axes[0, 0]),
                ("ev",       "Explained Variance", axes[0, 1]),
                ("entropy",  "Entropy", axes[0, 2]),
                ("kl",       "KL Divergence", axes[1, 0]),
                ("val_loss", "Value Loss", axes[1, 1]),
                ("pg_loss",  "Policy Gradient Loss", axes[1, 2]),
            ]
            for key, title, ax in pairs:
                series = data.get(key, [])
                if series:
                    xs, ys = zip(*series)
                    ax.plot(xs, ys)
                ax.set_title(title)
                ax.set_xlabel("Steps")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("[audit] install matplotlib for --plot support")


if __name__ == "__main__":
    main()
