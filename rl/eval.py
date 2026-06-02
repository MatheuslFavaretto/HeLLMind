"""Deterministic evaluation of a saved brain — the rigorous way to test performance.

Training metrics are noisy (exploration + reward shaping). This loads a checkpoint and
runs N episodes with `deterministic=True` (no exploration), reporting clean numbers:
mean reward, shooting accuracy, kills/episode, success rate. Use it to:
  - measure a brain's real performance, and
  - A/B two brains (same task, change one thing) by comparing their eval numbers.

    python -m rl.eval                      # evaluate this vault's brain (20 episodes)
    python -m rl.eval --episodes 50
    python -m rl.eval --path ./checkpoints/ppo_defend_the_center_a3_final.zip
"""
import argparse

import numpy as np
from stable_baselines3 import PPO

from config import Config
from doom.campaign import campaign_metadata
from doom.env import probe_env_metadata
from instrumentation.stats_tracker import StatsTracker
from rl.train import _latest_checkpoint, build_vec_env


def evaluate(cfg: Config, path: str, button_names: list, episodes: int = 20) -> dict:
    """Run `episodes` deterministic episodes and return a metrics summary."""
    venv = build_vec_env(cfg)  # n_envs forced to 1 by the caller
    model = PPO.load(path, env=venv)
    tracker = StatsTracker(button_names=button_names)

    obs = venv.reset()
    done_count = 0
    while done_count < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _rewards, _dones, infos = venv.step(action)
        tracker.update(infos, np.asarray(action))
        done_count += sum(1 for i in infos if i.get("episode"))
    venv.close()
    return tracker.snapshot(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministically evaluate a saved brain.")
    p.add_argument("--episodes", type=int, default=20, help="Episodes to run.")
    p.add_argument("--path", default=None, help="Checkpoint .zip (default: vault's latest).")
    p.add_argument("--render", action="store_true", help="Show the Doom window.")
    args = p.parse_args()

    cfg = Config()
    cfg.n_envs = 1            # single env for a clean, reproducible eval
    cfg.docs_enabled = False  # no LLM/notes during eval
    cfg.memory_enabled = False
    if args.render:
        cfg.render = True

    if cfg.campaign:
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0])
        name_prefix = f"ppo_campaign_a{meta['num_actions']}"
    else:
        meta = probe_env_metadata(cfg.scenario, cfg.frame_skip, cfg.resolution)
        name_prefix = f"ppo_{cfg.scenario}_a{meta['num_actions']}"
    button_names = meta["button_names"]

    path = args.path or _latest_checkpoint(cfg, name_prefix)
    if not path:
        raise SystemExit(f"No checkpoint found for '{name_prefix}' in {cfg.checkpoint_dir}. "
                         "Train first, or pass --path.")
    print(f"[eval] {path} | {args.episodes} deterministic episodes")

    s = evaluate(cfg, path, button_names, args.episodes)
    print("\n== Evaluation ==")
    print(f"  episodes:        {int(s['episodes'])}")
    print(f"  RAW reward/ep:   {s['mean_base_reward']:.2f}   (native scenario, fair for A/B)")
    print(f"  shaped reward/ep:{s['mean_reward']:.2f}   (includes reward shaping)")
    print(f"  shooting acc.:   {s['shooting_accuracy']:.0%}")
    print(f"  kills/episode:   {s['kills_per_episode']:.2f}")
    print(f"  success rate:    {s['success_rate']:.0%}")
    print(f"  mean ep length:  {s['mean_episode_length']:.0f} steps")


if __name__ == "__main__":
    main()
