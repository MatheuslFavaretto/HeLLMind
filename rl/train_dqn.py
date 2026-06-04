"""QR-DQN training engine (V2 Phase 1).

Off-policy, distributional Q-learning with a replay buffer — the sample-efficient
alternative to PPO. Key differences vs train.py (PPO):

  • Replay buffer: learns from past experiences, not just fresh rollouts.
  • Continuous updates: updates the policy every N steps (not every rollout).
  • More sample-efficient: learns more per environment step (critical for Doom).
  • No on-policy constraints: safe to use small batch sizes + frequent updates.

Memory math (so the buffer doesn't OOM a 16 GB machine):
  Dict obs: 84×84×channels uint8 + game_vars float32 ≈ 14 KB/step
  buffer_size=50_000 → ~700 MB obs (×2 for next_obs) → ~1.5 GB total. Safe.

References:
  sb3_contrib.QRDQN (QR-DQN: Quantile Regression DQN, Dabney et al. 2017)
  cleanrl dqn_atari.py — single-file clarity this module mirrors
"""
import argparse
import glob
import os
import sys
from typing import Optional

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _latest_dqn_checkpoint(checkpoint_dir: str, prefix: str) -> Optional[str]:
    cands = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*.zip"))
    return max(cands, key=os.path.getmtime) if cands else None


def _dqn_prefix(n_actions: int, game_vars: bool) -> str:
    gv = "_gv" if game_vars else ""
    return f"qrdqn_campaign_a{n_actions}{gv}"


# ──────────────────────────────────────────────────────────────────────────────
# Build env
# ──────────────────────────────────────────────────────────────────────────────

def build_env(cfg, doom_map: str, n_envs: int = 1):
    """DQN is typically single-env (the replay buffer handles diversity). We support
    n_envs > 1 via SubprocVecEnv + the SB3 multi-env DQN path."""
    from doom.campaign import make_campaign_env
    fns = [make_campaign_env(cfg, doom_map, rank,
                             memory_dir=cfg.memory_dir if cfg.memory_enabled else None)
           for rank in range(n_envs)]
    venv = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)
    return VecMonitor(venv)


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg, doom_map: str, timesteps: int, fresh: bool = False,
          n_envs: int = 1, verbose: int = 1) -> str:
    """Train (or resume) a QR-DQN agent. Returns the final checkpoint path."""
    from sb3_contrib import QRDQN
    from doom.campaign import campaign_metadata

    meta = campaign_metadata(cfg.wad_path, doom_map, strafe=cfg.strafe)
    prefix = _dqn_prefix(meta["num_actions"], cfg.game_vars)
    ck_dir = cfg.checkpoint_dir
    os.makedirs(ck_dir, exist_ok=True)
    device = _best_device()

    venv = build_env(cfg, doom_map, n_envs)

    # Buffer size: capped so we stay inside RAM.
    # 14 KB/step × 50k = ~700 MB obs; ×2 (next_obs) + overhead ≈ 1.5 GB.
    buffer_size = int(os.getenv("DQN_BUFFER", "50000"))
    batch_size  = int(os.getenv("DQN_BATCH",  "32"))
    lr          = float(os.getenv("DQN_LR",   "1e-4"))
    # Start learning only after the buffer has some diversity; τ for target-net soft update.
    learning_starts = int(os.getenv("DQN_WARMUP", "5000"))
    tau             = float(os.getenv("DQN_TAU",   "1.0"))    # 1.0 = hard update (periodic)
    target_update   = int(os.getenv("DQN_TARGET",  "1000"))   # steps between target updates
    train_freq      = int(os.getenv("DQN_TRAIN_FREQ", "4"))   # update every N steps
    n_quantiles     = int(os.getenv("DQN_QUANTILES", "200"))  # QR-DQN distributional atoms

    resume_path = None if fresh else _latest_dqn_checkpoint(ck_dir, prefix)
    if resume_path:
        print(f"[qrdqn] resuming: {os.path.basename(resume_path)}  [device={device}]")
        model = QRDQN.load(resume_path, env=venv, device=device,
                           tensorboard_log=cfg.tensorboard_log)
        reset_ts = False
    else:
        print(f"[qrdqn] new brain  prefix={prefix}  device={device}  "
              f"buffer={buffer_size:,}  batch={batch_size}  lr={lr}")
        model = QRDQN(
            policy="MultiInputPolicy" if cfg.game_vars else "CnnPolicy",
            env=venv,
            learning_rate=lr,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=cfg.gamma,
            train_freq=train_freq,
            gradient_steps=1,
            target_update_interval=target_update,
            exploration_fraction=0.1,   # ε-greedy: explore for 10% of total steps
            exploration_final_eps=0.05,
            policy_kwargs={"n_quantiles": n_quantiles},
            device=device,
            seed=cfg.seed,
            tensorboard_log=cfg.tensorboard_log,
            verbose=verbose,
        )
        reset_ts = True

    cb = CheckpointCallback(
        save_freq=max(cfg.write_every_steps // n_envs, 1),
        save_path=ck_dir, name_prefix=prefix)

    model.learn(total_timesteps=timesteps, callback=cb,
                reset_num_timesteps=reset_ts, progress_bar=True)

    final = os.path.join(ck_dir, f"{prefix}_final.zip")
    model.save(final)
    print(f"[qrdqn] saved → {final}")
    venv.close()
    return final


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="QR-DQN training (V2 off-policy engine).")
    p.add_argument("--map", default=None)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    from config import Config
    cfg = Config()
    doom_map = args.map or cfg.maps[0]
    train(cfg, doom_map, args.timesteps, args.fresh, args.n_envs, args.verbose)


if __name__ == "__main__":
    main()
