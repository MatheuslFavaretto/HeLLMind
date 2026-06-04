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
import time
from typing import Optional

import numpy as np

from config import Config
from doom.campaign import campaign_metadata
from doom.env import probe_env_metadata
from instrumentation.stats_tracker import StatsTracker
from rl.train import _latest_checkpoint, build_vec_env


def _tempered_actions(model, obs, temperature: float) -> np.ndarray:
    """Sample discrete actions from the policy with its logits scaled by 1/temperature.

    temperature → 0  ≈ argmax (deterministic),  1.0 = normal sampling,  >1 = flatter/random.
    A value like 0.5 sharpens toward the best actions while keeping enough stochasticity to
    avoid the argmax-collapse where the single most-probable action is a bad one."""
    import torch
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        logits = dist.distribution.logits          # Categorical (discrete action space)
        tempered = torch.distributions.Categorical(logits=logits / max(temperature, 1e-6))
        actions = tempered.sample()
    return actions.cpu().numpy()


def evaluate(cfg: Config, path: str, button_names: list, episodes: int = 20,
             deterministic: bool = True, temperature: Optional[float] = None) -> dict:
    """Run `episodes` and return a metrics summary.

    `deterministic=True` (default) takes the argmax action — the honest measure of what the
    brain has *committed* to. `deterministic=False` samples the policy at temperature 1.0.
    `temperature=τ` (overrides both) samples with logits scaled by 1/τ — a tunable middle
    ground that lets a brain whose argmax collapsed still act on its learned distribution
    (e.g. τ=0.5 explores without freezing). Feed-forward policies only."""
    venv = build_vec_env(cfg)  # n_envs forced to 1 by the caller
    from rl.algo import algo_class
    # Detect a recurrent brain from its tagged name so eval "just works" on any checkpoint,
    # even when the caller forgot to set USE_LSTM (a feed-forward load would crash on it).
    import os
    use_lstm = cfg.use_lstm or "_lstm" in os.path.basename(path)
    if temperature is not None and use_lstm:
        print("[eval] temperature sampling is feed-forward only; ignoring for the LSTM brain.")
        temperature = None
    model = algo_class(use_lstm).load(path, env=venv)
    tracker = StatsTracker(button_names=button_names)
    # When rendering, throttle to ~real time so the window is actually watchable
    # (otherwise ViZDoom blasts through hundreds of fps and the episodes flash by).
    step_delay = (cfg.frame_skip / 35.0) if cfg.render else 0.0

    obs = venv.reset()
    done_count = 0
    # Recurrent-safe loop: carry the LSTM hidden state and flag episode boundaries so it
    # resets per episode. For feed-forward PPO these args are simply unused (state stays
    # None), so the same loop drives both policies.
    lstm_states = None
    episode_starts = np.ones((venv.num_envs,), dtype=bool)
    while done_count < episodes:
        if temperature is not None:
            action = _tempered_actions(model, obs, temperature)
        else:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts,
                deterministic=deterministic)
        obs, _rewards, dones, infos = venv.step(action)
        episode_starts = dones
        tracker.update(infos, np.asarray(action))
        done_count += sum(1 for i in infos if i.get("episode"))
        if step_delay:
            time.sleep(step_delay)
    venv.close()
    return tracker.snapshot(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministically evaluate a saved brain.")
    p.add_argument("--episodes", type=int, default=20, help="Episodes to run.")
    p.add_argument("--path", default=None, help="Checkpoint .zip (default: vault's latest).")
    p.add_argument("--render", action="store_true", help="Show the Doom window.")
    p.add_argument("--json", action="store_true",
                   help="Also print a one-line JSON of the key metrics (for the supervisor).")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample the policy instead of argmax — reveals what an unconverged "
                        "brain has learned but can't yet argmax (e.g. fights when sampled).")
    p.add_argument("--temperature", type=float, default=None,
                   help="Tempered sampling (e.g. 0.5): scales logits by 1/T. Sharpens toward "
                        "the best actions but avoids the argmax-collapse. Overrides --stochastic.")
    args = p.parse_args()

    cfg = Config()
    cfg.n_envs = 1            # single env for a clean, reproducible eval
    cfg.docs_enabled = False  # no LLM/notes during eval
    cfg.memory_enabled = False
    if args.render:
        cfg.render = True

    from rl.algo import brain_prefix
    if cfg.campaign:
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0], strafe=cfg.strafe)
        name_prefix = brain_prefix("campaign", meta["num_actions"], cfg.use_lstm,
                                   cfg.spatial_memory, cfg.depth_perception, cfg.automap, cfg.frame_stack, cfg.game_vars)
    else:
        meta = probe_env_metadata(cfg.scenario, cfg.frame_skip, cfg.resolution)
        name_prefix = brain_prefix(cfg.scenario, meta["num_actions"], cfg.use_lstm,
                                   cfg.spatial_memory, cfg.depth_perception, cfg.automap, cfg.frame_stack, cfg.game_vars)
    button_names = meta["button_names"]

    path = args.path or _latest_checkpoint(cfg, name_prefix)
    if not path:
        raise SystemExit(f"No checkpoint found for '{name_prefix}' in {cfg.checkpoint_dir}. "
                         "Train first, or pass --path.")
    if args.temperature is not None:
        mode = f"tempered sampling (T={args.temperature})"
    elif args.stochastic:
        mode = "stochastic (sampled)"
    else:
        mode = "deterministic (argmax)"
    print(f"[eval] {path} | {args.episodes} {mode} episodes")

    s = evaluate(cfg, path, button_names, args.episodes,
                 deterministic=not args.stochastic, temperature=args.temperature)
    print("\n== Evaluation ==")
    print(f"  episodes:        {int(s['episodes'])}")
    print(f"  RAW reward/ep:   {s['mean_base_reward']:.2f}   (native scenario, fair for A/B)")
    print(f"  shaped reward/ep:{s['mean_reward']:.2f}   (includes reward shaping)")
    print(f"  shooting acc.:   {s['shooting_accuracy']:.0%}")
    print(f"  kills/episode:   {s['kills_per_episode']:.2f}")
    print(f"  success rate:    {s['success_rate']:.0%}")
    print(f"  exit rate:       {s.get('exit_rate', 0.0):.0%}   (reached the level end)")
    cov = s.get("map_coverage", {}) or {}
    print(f"  map explored:    {cov.get('explored_fraction', 0.0):.0%}   "
          f"({int(cov.get('cells_visited', 0))} cells)")
    if s.get("terminals"):
        print(f"  episode endings: {s['terminals']}")
    print(f"  mean ep length:  {s['mean_episode_length']:.0f} steps")

    if args.json:
        import json
        cov = s.get("map_coverage", {}) or {}
        n_eps = max(s.get("episodes", 1), 1)
        terminals = s.get("terminals", {})
        metrics = {
            "kills_per_episode": float(s["kills_per_episode"]),
            "shooting_accuracy": float(s["shooting_accuracy"]),
            "success_rate": float(s["success_rate"]),
            "exit_rate": float(s.get("exit_rate", 0.0)),
            "timeout_rate": float(terminals.get("timeout", 0)) / n_eps,
            "death_rate": float(terminals.get("death", 0)) / n_eps,
            "explored_fraction": float(cov.get("explored_fraction", 0.0)),
            "cells_visited": float(cov.get("cells_visited", 0.0)),
            "mean_base_reward": float(s["mean_base_reward"]),
            "mean_episode_length": float(s["mean_episode_length"]),
        }
        print("METRICS_JSON " + json.dumps(metrics))


if __name__ == "__main__":
    main()
