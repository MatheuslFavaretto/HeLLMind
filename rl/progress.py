"""Prove the brain is REALLY learning: deterministically evaluate several saved
checkpoints and show the progression. Training curves (ep_rew_mean) are noisy and can
mislead; the honest test is whether the ARGMAX policy improves over checkpoints.

    python -m rl.progress --episodes 8 --points 5     # eval 5 evenly-spaced checkpoints
"""
import argparse
import glob
import os
import re

from config import Config
from doom.campaign import campaign_metadata
from doom.env import probe_env_metadata
from rl.algo import brain_prefix
from rl.eval import evaluate

_STEP = re.compile(r"_(\d+)_steps\.zip$")


def _checkpoints(cfg: Config, name_prefix: str):
    out = []
    for f in glob.glob(os.path.join(cfg.checkpoint_dir, f"{name_prefix}_*_steps.zip")):
        m = _STEP.search(f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Show the deterministic-eval progression across checkpoints.")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--points", type=int, default=5, help="How many checkpoints to sample.")
    args = p.parse_args()

    cfg = Config()
    cfg.n_envs = 1
    cfg.docs_enabled = False
    cfg.memory_enabled = False
    if cfg.campaign:
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0], strafe=cfg.strafe)
        name_prefix = brain_prefix("campaign", meta["num_actions"], cfg.use_lstm,
                                   cfg.spatial_memory, cfg.depth_perception, cfg.automap, cfg.frame_stack, cfg.game_vars)
    else:
        meta = probe_env_metadata(cfg.scenario, cfg.frame_skip, cfg.resolution)
        name_prefix = brain_prefix(cfg.scenario, meta["num_actions"], cfg.use_lstm,
                                   cfg.spatial_memory, cfg.depth_perception, cfg.automap, cfg.frame_stack, cfg.game_vars)

    ckpts = _checkpoints(cfg, name_prefix)
    if len(ckpts) < 2:
        raise SystemExit(
            f"Need ≥2 checkpoints for a progression (found {len(ckpts)} for '{name_prefix}').\n"
            "Train with WRITE_EVERY_STEPS set so intermediate checkpoints are kept.")
    # evenly sample `points` checkpoints across the run
    if len(ckpts) > args.points:
        idx = [round(i * (len(ckpts) - 1) / (args.points - 1)) for i in range(args.points)]
        ckpts = [ckpts[i] for i in sorted(set(idx))]

    print(f"Deterministic-eval progression ({args.episodes} eps each) — the honest learning signal\n")
    print(f"{'steps':>9} | {'kills/ep':>8} | {'accuracy':>8} | {'explored':>8} | {'success':>7}")
    print("-" * 52)
    prev = None
    for steps, path in ckpts:
        s = evaluate(cfg, path, meta["button_names"], args.episodes)
        cov = s.get("map_coverage", {}) or {}
        k = s["kills_per_episode"]
        arrow = "" if prev is None else (" ↑" if k > prev else " ↓" if k < prev else " =")
        print(f"{steps:>9,} | {k:>8.2f} | {s['shooting_accuracy']:>7.0%} | "
              f"{cov.get('explored_fraction', 0.0):>7.0%} | {s['success_rate']:>6.0%}{arrow}")
        prev = k
    print("\nRising kills/accuracy across checkpoints = the policy is genuinely learning.")


if __name__ == "__main__":
    main()
