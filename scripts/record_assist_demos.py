"""Record demos from the ASSISTS playing — the "teacher" half of teacher→student→solo.

The hand-coded assists (auto-aim, vision door-nav, best-weapon, USE) are a decent scripted
expert. We let them DRIVE the campaign env and log, every frame, (pixel observation → the action
the ASSIST actually executed). Those (obs, action) pairs are saved as standard BC demos, so
`python -m rl.bc --demos <dir>` can CLONE the assists' skill into the network — which then runs
SOLO (assists off). This is imitation learning: the network learns to copy the teacher.

Why this and not "train PPO with assists on": PPO logs the NETWORK's action but the env executes
the ASSIST's — the reward credits the wrong action and the gradient is corrupted. Cloning targets
the assist's executed action directly (supervised), so it actually transfers.

    python scripts/record_assist_demos.py --episodes 8        # -> vault/.memory/assist_demos/
    python -m rl.bc --demos vault/.memory/assist_demos --only-success   # clone the wins
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from config import Config  # noqa: E402
from doom.campaign import (CAMPAIGN_BUTTONS, campaign_actions,  # noqa: E402
                           make_campaign_env)
from rl.bc import nearest_action, save_demo  # noqa: E402


def _action_vectors(strafe: bool):
    """(button-vector) per discrete action, aligned to CAMPAIGN_BUTTONS — for nearest_action."""
    bidx = {b.name: i for i, b in enumerate(CAMPAIGN_BUTTONS)}
    vecs = []
    for combo, _label in campaign_actions(strafe):
        v = [0] * len(CAMPAIGN_BUTTONS)
        for name in combo:
            v[bidx[name]] = 1
        vecs.append(v)
    return vecs


def main() -> None:
    p = argparse.ArgumentParser(description="Record demos from the assists (teacher) playing.")
    p.add_argument("--episodes", type=int, default=8, help="Episodes to record.")
    p.add_argument("--out", default=None, help="Demos dir (default <vault>/.memory/assist_demos).")
    p.add_argument("--map", default=None, help="Map to record on (default: cfg.maps[0]).")
    args = p.parse_args()

    cfg = Config()
    cfg.n_envs = 1
    cfg.render = False
    cfg.docs_enabled = False
    cfg.memory_enabled = False
    # The teacher = ALL assists ON (they produce the expert actions we clone).
    cfg.auto_aim = cfg.auto_door_nav = cfg.auto_best_weapon = cfg.auto_use = True
    # Semantic channel is irrelevant here (we record raw pixels); keep it off for speed.
    cfg.semantic_channel = False
    doom_map = args.map or cfg.maps[0]
    out_dir = args.out or os.path.join(cfg.memory_dir, "assist_demos")
    os.makedirs(out_dir, exist_ok=True)

    action_vecs = _action_vectors(cfg.strafe)
    env = make_campaign_env(cfg, doom_map, 0, rewards=cfg.reward_weights(),
                            memory_dir=None)()
    print(f"[teacher] recording {args.episodes} assisted episodes on {doom_map} -> {out_dir}")

    n_exit = 0
    for ep in range(args.episodes):
        env.reset(seed=cfg.seed + ep)
        frames, acts = [], []
        terminal = "timeout"
        for _ in range(cfg.episode_timeout + 1):
            st = env.game.get_state()
            if st is None:
                break
            frame = cv2.resize(st.screen_buffer, (env.width, env.height),
                               interpolation=cv2.INTER_AREA)
            frames.append(frame[:, :, None])               # [H,W,1] pixel obs, like human demos
            # Drive with a neutral base action; the assists override it with the expert move.
            _obs, _r, done, _t, info = env.step(0)
            executed = env.game.get_last_action()          # buttons ACTUALLY executed (post-assist)
            acts.append(nearest_action(executed, action_vecs))
            if done:
                terminal = (info.get("doom", {}) or {}).get("terminal", "timeout")
                break
        reached_exit = (terminal == "exit")
        n_exit += int(reached_exit)
        if frames:
            path = os.path.join(out_dir, f"assist_{doom_map}_{ep:03d}.npz")
            save_demo(path, np.asarray(frames, dtype=np.uint8),
                      np.asarray(acts, dtype=np.int64), reached_exit=reached_exit)
            tag = "✓ EXIT" if reached_exit else f"✗ {terminal}"
            print(f"  ep {ep}: {len(acts)} steps  {tag}  -> {os.path.basename(path)}")
    env.close()
    print(f"[teacher] done. {n_exit}/{args.episodes} reached the exit. "
          f"Clone with: python -m rl.bc --demos {out_dir}"
          + ("  --only-success" if n_exit else ""))


if __name__ == "__main__":
    main()
