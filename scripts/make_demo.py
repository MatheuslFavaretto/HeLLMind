"""Render a high-res gameplay GIF + stills from the ACTUAL game screen (640x480 RGB),
not the 84x84 policy input — for the README. Headless (no window). The policy still sees
the same 84x84 grayscale 4-frame stack it trained on; we just capture the pretty buffer.

    python make_demo.py --map MAP02 --steps 240 --out assets/gameplay.gif
"""
import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import vizdoom as vzd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from doom.campaign import CAMPAIGN_ACTIONS, CAMPAIGN_BUTTONS, default_wad
from rl.algo import algo_class, brain_prefix
from rl.train import _latest_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="MAP02")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--out", default="assets/gameplay.gif")
    p.add_argument("--path", default=None)
    args = p.parse_args()

    cfg = Config()
    name_prefix = brain_prefix("campaign", len(CAMPAIGN_ACTIONS), cfg.use_lstm,
                               cfg.spatial_memory, cfg.depth_perception, cfg.automap, cfg.frame_stack, cfg.game_vars)
    path = args.path or _latest_checkpoint(cfg, name_prefix)
    model = algo_class(cfg.use_lstm).load(path)
    print(f"[demo] brain: {path}")

    game = vzd.DoomGame()
    game.set_doom_game_path(default_wad())
    game.set_doom_map(args.map)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.set_available_buttons(CAMPAIGN_BUTTONS)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(4200)
    game.init()
    game.new_episode()

    bidx = {b.name: i for i, b in enumerate(CAMPAIGN_BUTTONS)}
    actions = []
    for combo, _label in CAMPAIGN_ACTIONS:
        v = [0] * len(CAMPAIGN_BUTTONS)
        for n in combo:
            v[bidx[n]] = 1
        actions.append(v)

    stack: deque = deque(maxlen=cfg.frame_stack)
    frames = []
    for _ in range(args.steps):
        if game.is_episode_finished():
            game.new_episode()
            stack.clear()
        rgb = game.get_state().screen_buffer            # (480, 640, 3)
        small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (84, 84),
                           interpolation=cv2.INTER_AREA)
        while len(stack) < cfg.frame_stack:
            stack.append(small)
        stack.append(small)
        obs = np.stack(list(stack), axis=2)[None]       # (1, 84, 84, 4) -> predict transposes
        action, _ = model.predict(obs, deterministic=True)
        game.make_action(actions[int(np.asarray(action).flat[0])], cfg.frame_skip)
        frames.append(Image.fromarray(rgb))
    game.close()

    # Full-res stills for the README.
    for i, fi in enumerate((60, 140, 210)):
        if fi < len(frames):
            frames[fi].save(args.out.rsplit(".", 1)[0] + f"-still{i + 1}.png")
    # Smaller, palette-quantized GIF so it's README-friendly (a few MB, not ~18).
    small = [f.resize((320, 240)).convert("P", palette=Image.ADAPTIVE, colors=128)
             for f in frames[::2]]                       # half the frames, half the size
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    small[0].save(args.out, save_all=True, append_images=small[1:],
                  duration=110, loop=0, optimize=True)
    print(f"[demo] wrote {args.out} ({len(small)} frames) + {len(frames)//80} stills")


if __name__ == "__main__":
    main()
