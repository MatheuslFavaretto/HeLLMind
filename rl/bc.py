"""Behavioral cloning from human SPECTATOR demos — bootstrap the agent with a real success.

The exit is a near-impossible accidental discovery (reward is ~zero until the agent first
stumbles onto it). The classic fix is to SHOW it once: a human plays a few episodes that
reach the exit, we record (observation, action) pairs, and supervised-train the policy to
imitate them. The RL run then *continues* from a brain that already knows roughly how to get
there — turning an impossible exploration problem into a fine-tuning one.

Pipeline:
    1. record  — `scripts/record_demo.py` (ViZDoom SPECTATOR; a human plays, we log).
    2. clone   — `python -m rl.bc --demos <dir>` supervised-trains the PPO policy on them.
    3. train   — `doom-cli train --resume` continues with RL from the cloned brain.

The pure pieces (button→action mapping, demo IO, the BC loss) are unit-tested; building the
SB3 policy + the supervised loop reuse the same network the RL uses.
"""
import argparse
import glob
import os
from typing import Optional, Sequence, Tuple

import numpy as np


def nearest_action(pressed: Sequence[int], actions: Sequence[Sequence[int]]) -> int:
    """Map a raw pressed-button vector (what a human held) to the closest discrete COMBINED
    action index. Exact match wins; otherwise the action sharing the most buttons (and adding
    the fewest extra) — so 'forward+attack' held maps to the FWD+ATK combo, not bare ATK."""
    pressed = [1 if b else 0 for b in pressed]
    best_i, best_score = 0, -1e9
    for i, act in enumerate(actions):
        act = [1 if b else 0 for b in act]
        overlap = sum(p and a for p, a in zip(pressed, act))
        missing = sum(p and not a for p, a in zip(pressed, act))   # human pressed, action lacks
        extra = sum(a and not p for p, a in zip(pressed, act))     # action adds, human didn't
        score = 2 * overlap - missing - extra                      # reward overlap, punish mismatch
        if score > best_score:
            best_i, best_score = i, score
    return best_i


# ---------------------------------------------------------------------------
# Demo IO  (one .npz per episode: obs uint8 [N,H,W,C], actions int [N])
# ---------------------------------------------------------------------------

def save_demo(path: str, obs: np.ndarray, actions: np.ndarray,
              reached_exit: Optional[bool] = None) -> None:
    """Save one episode's (obs, action) pairs. `reached_exit` flags whether the human
    actually reached the level exit — BC on demos that WANDER or DIE clones wandering/dying,
    so recording this lets us train only on the successful runs (the whole point of BC)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = {} if reached_exit is None else {"reached_exit": np.array(bool(reached_exit))}
    np.savez_compressed(path, obs=obs.astype(np.uint8),
                        actions=actions.astype(np.int64), **extra)


def load_demos(demos_dir: str, only_success: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate every .npz demo in a dir into (obs, actions). Empty -> (0-len arrays).

    only_success=True keeps ONLY demos flagged reached_exit=True (BC's whole premise: imitate
    a SUCCESS). Demos with no flag (older recordings) are treated as unknown and kept, with a
    warning — re-record them with the updated record_demo to get the flag."""
    files = sorted(glob.glob(os.path.join(demos_dir, "*.npz")))
    obs_chunks, act_chunks = [], []
    kept, skipped, unflagged = 0, 0, 0
    for f in files:
        with np.load(f) as d:
            if "obs" not in d or "actions" not in d or len(d["actions"]) == 0:
                continue
            if only_success and "reached_exit" in d:
                if not bool(d["reached_exit"]):
                    skipped += 1
                    continue
            elif only_success:
                unflagged += 1  # no flag → can't verify; keep but warn
            obs_chunks.append(d["obs"])
            act_chunks.append(d["actions"])
            kept += 1
    if only_success:
        msg = f"[bc] using {kept} demo(s)"
        if skipped:
            msg += f", skipped {skipped} that didn't reach the exit"
        if unflagged:
            msg += f"; {unflagged} have no exit flag (re-record for verification)"
        print(msg)
    if not obs_chunks:
        return np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    return np.concatenate(obs_chunks, axis=0), np.concatenate(act_chunks, axis=0)


def bc_cross_entropy(logits, targets):
    """Behavioral-cloning loss: cross-entropy of the policy's action logits vs the human's
    actions. Pure torch so it's unit-testable without ViZDoom."""
    import torch.nn.functional as F
    return F.cross_entropy(logits, targets)


def preprocess_demo_batch(frames_uint8, cfg, obs_space, device):
    """Convert a batch of recorded PIXEL frames [B,H,W,C_rec] into the policy's obs tensor:
    channels-first, padded to base_ch (zeros for unrecorded spatial/depth), tiled across the
    frame stack, and wrapped as a Dict {image, vars} with NEUTRAL vars when game_vars is on.
    Shared by BC training (behavioral_clone) and BC-regularized fine-tune (rl.bc_finetune)."""
    import numpy as _np
    import torch as _th
    if cfg.game_vars:
        img_space, vars_space = obs_space["image"], obs_space["vars"]
    else:
        img_space, vars_space = obs_space, None
    c_total = int(img_space.shape[0])           # base_ch × frame_stack
    base_ch = max(1, c_total // cfg.frame_stack)
    frames = _np.asarray(frames_uint8, dtype=_np.float32)
    frames = _np.transpose(frames, (0, 3, 1, 2))            # [B,C_rec,H,W]
    if frames.shape[1] < base_ch:
        pad = _np.zeros((frames.shape[0], base_ch - frames.shape[1],
                         frames.shape[2], frames.shape[3]), dtype=frames.dtype)
        frames = _np.concatenate([frames, pad], axis=1)
    elif frames.shape[1] > base_ch:
        frames = frames[:, :base_ch]
    stacked = _np.tile(frames, (1, cfg.frame_stack, 1, 1))  # [B, base_ch*stack, H, W]
    img_t = _th.as_tensor(stacked, device=device)
    if vars_space is not None:
        neutral = _np.ones((frames.shape[0], *vars_space.shape), dtype=_np.float32)
        return {"image": img_t, "vars": _th.as_tensor(neutral, device=device)}
    return img_t


# ---------------------------------------------------------------------------
# Orchestration  (build the SB3 policy, supervised-train it, save the brain)
# ---------------------------------------------------------------------------

def behavioral_clone(cfg, demos_dir: str, epochs: int = 10, batch_size: int = 64,
                     lr: float = 3e-4, only_success: bool = False) -> Optional[str]:
    """Supervised-train the PPO policy to imitate the demos; save it as this vault's brain
    so `train --resume` continues from it. Returns the saved path (or None if no demos).

    only_success=True trains ONLY on demos that reached the exit (BC's premise: clone a win)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

    from doom.campaign import campaign_metadata, make_campaign_env
    from rl.algo import brain_prefix

    obs, actions = load_demos(demos_dir, only_success=only_success)
    if len(actions) == 0:
        print(f"[bc] no demos found in {demos_dir}")
        return None
    print(f"[bc] {len(actions)} (obs, action) pairs from {demos_dir}")

    # Build a vec env matching the training obs pipeline, so the policy shape is identical.
    from rl.algo import policy_name
    venv = DummyVecEnv([make_campaign_env(cfg, cfg.maps[0], 0)])
    venv = VecFrameStack(VecTransposeImage(venv), n_stack=cfg.frame_stack)
    model = PPO(policy_name(cfg.use_lstm, cfg.game_vars), venv, verbose=0, seed=cfg.seed)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    device = policy.device

    # Match the recorded obs to the policy input. The recorder captures only PIXELS
    # ([N,H,W,C_rec], usually 1 channel); the policy expects framestacked channels-first
    # [base_ch × frame_stack, H, W], where base_ch may be >1 (spatial/depth/automap). We:
    #   1) pad the missing base channels with ZEROS (an unrecorded spatial grid starts empty
    #      anyway — the human demo teaches pixels→action; RL learns the extra channels later),
    #   2) TILE the frame across the stack (not repeat — tile preserves the [..ch0,ch1..]
    #      per-frame ordering VecFrameStack uses).
    # The policy obs may be an image Box OR a Dict {image, vars} (game_vars). Resolve the
    # image sub-space and the vars sub-space (if any).
    ospace = model.observation_space
    n = len(actions)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        total = 0.0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            obs_t = preprocess_demo_batch(obs[b], cfg, ospace, device)
            act_t = torch.as_tensor(actions[b], device=device).long()
            dist = policy.get_distribution(obs_t)
            logits = dist.distribution.logits
            loss = bc_cross_entropy(logits, act_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        print(f"[bc] epoch {ep + 1}/{epochs}  loss={total / n:.4f}")

    meta = campaign_metadata(cfg.wad_path, cfg.maps[0], strafe=cfg.strafe)
    name_prefix = brain_prefix("campaign", meta["num_actions"], cfg.use_lstm,
                               cfg.spatial_memory, cfg.depth_perception, cfg.automap,
                               cfg.frame_stack, cfg.game_vars,
                               getattr(cfg, "semantic_channel", False))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    out = os.path.join(cfg.checkpoint_dir, f"{name_prefix}_final.zip")
    model.save(out)
    venv.close()
    print(f"[bc] saved cloned brain -> {out}")
    return out


def main() -> None:
    from config import Config
    p = argparse.ArgumentParser(description="Behavioral cloning from human demos.")
    p.add_argument("--demos", default=None, help="Demos dir (default: <memory>/demos).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--only-success", action="store_true",
                   help="Train ONLY on demos that reached the exit (BC's premise: clone a win).")
    args = p.parse_args()
    cfg = Config()
    demos_dir = args.demos or os.path.join(cfg.memory_dir, "demos")
    behavioral_clone(cfg, demos_dir, epochs=args.epochs, batch_size=args.batch_size,
                     only_success=args.only_success)


if __name__ == "__main__":
    main()
