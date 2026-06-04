"""Render a gameplay GIF + screenshots from a trained brain, straight from the
observation tensor (no screen recording needed). For the spatial-memory brain it shows,
side by side: what the agent SEES and the agent's MEMORY of where it has been."""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from doom.campaign import campaign_metadata
from rl.algo import algo_class, brain_prefix
from rl.train import _latest_checkpoint, build_vec_env

UP = 4  # upscale factor for the tiny 84x84 view


def _tile(frame: np.ndarray, visit) -> Image.Image:
    """Upscale the agent view (and optional memory) into one RGB frame."""
    def up(a):
        img = Image.fromarray(a.astype(np.uint8), mode="L").resize(
            (84 * UP, 84 * UP), Image.NEAREST
        )
        return img.convert("RGB")
    view = up(frame)
    if visit is None:
        return view
    # Tint the memory channel green so it reads as "where I've been".
    mem = np.zeros((*visit.shape, 3), dtype=np.uint8)
    mem[..., 1] = visit  # green
    mem_img = Image.fromarray(mem).resize((84 * UP, 84 * UP), Image.NEAREST)
    combo = Image.new("RGB", (84 * UP * 2 + 8, 84 * UP), "black")
    combo.paste(view, (0, 0))
    combo.paste(mem_img, (84 * UP + 8, 0))
    return combo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=None)
    p.add_argument("--steps", type=int, default=320)
    p.add_argument("--out", default="./vault/attachments/gameplay.gif")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample the policy (vs argmax) — for unconverged brains.")
    args = p.parse_args()

    cfg = Config()
    cfg.n_envs = 1
    cfg.docs_enabled = False
    cfg.memory_enabled = False
    meta = campaign_metadata(cfg.wad_path, cfg.maps[0], strafe=cfg.strafe)
    name_prefix = brain_prefix("campaign", meta["num_actions"], cfg.use_lstm,
                               cfg.spatial_memory, cfg.depth_perception, cfg.automap, cfg.frame_stack, cfg.game_vars)
    path = args.path or _latest_checkpoint(cfg, name_prefix)
    print(f"[gif] brain: {path} | spatial={cfg.spatial_memory}")

    venv = build_vec_env(cfg)
    use_lstm = cfg.use_lstm or "_lstm" in os.path.basename(path or "")
    model = algo_class(use_lstm).load(path, env=venv)
    # Base channels in fixed order: pixels, [spatial], [depth], [automap]. The most-recent
    # frame block starts at (frame_stack-1)*base_ch; pixels is its first channel, spatial 2nd.
    base_ch = 1 + cfg.spatial_memory + cfg.depth_perception + cfg.automap
    f_idx = (cfg.frame_stack - 1) * base_ch
    v_idx = f_idx + 1 if cfg.spatial_memory else None

    obs = venv.reset()
    frames = []
    lstm_states = None
    starts = np.ones((venv.num_envs,), dtype=bool)
    for _ in range(args.steps):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=starts,
            deterministic=not args.stochastic)
        obs, _r, dones, _i = venv.step(action)
        starts = dones
        img = obs["image"] if isinstance(obs, dict) else obs  # game_vars -> Dict obs
        arr = np.asarray(img)[0]                    # (C, 84, 84)
        frame = arr[f_idx]
        visit = arr[v_idx] if v_idx is not None else None
        frames.append(_tile(frame, visit))
    venv.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True)
    # A few stills for the README.
    shots = os.path.join(os.path.dirname(args.out), "shot")
    for i, fi in enumerate((40, 120, 220)):
        if fi < len(frames):
            frames[fi].save(f"{shots}_{i+1}.png")
    print(f"[gif] wrote {args.out} ({len(frames)} frames) + screenshots")


if __name__ == "__main__":
    main()
