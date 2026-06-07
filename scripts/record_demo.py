"""Record human demonstrations in ViZDoom SPECTATOR mode (for behavioral cloning).

A human plays with the keyboard/mouse; we log (downsampled observation, nearest discrete
action) for every frame and save one .npz per episode — now WITH a `reached_exit` flag so
BC can clone ONLY the successful runs (wandering/dying demos teach the agent to wander/die).

The full BC → fine-tune pipeline (use a PIXEL-ONLY config so the brain's obs matches the
recorded pixel frames — spatial/depth/game-vars channels aren't recorded and would mismatch):

    # 1) record — play to the EXIT each episode (the flag is printed per episode)
    python scripts/record_demo.py --map MAP01 --episodes 5 --strafe --minutes 8

    # 2) clone — ONLY the runs that reached the exit, into a pixel-only brain
    CAMPAIGN=1 MAPS=MAP01 GAME_VARS=0 SPATIAL_MEMORY=0 DEPTH_PERCEPTION=0 AUTOMAP=0 STRAFE=1 \
      python -m rl.bc --epochs 20 --only-success

    # 3) fine-tune — RL continues FROM the cloned brain (same pixel-only config)
    CAMPAIGN=1 MAPS=MAP01 GAME_VARS=0 SPATIAL_MEMORY=0 DEPTH_PERCEPTION=0 AUTOMAP=0 STRAFE=1 \
      python -m rl.train --maps MAP01 --timesteps 800000 --resume

    # 4) eval
    …same env… python -m rl.eval --episodes 20 --json --algo ppo

Needs a display (it opens the game window). The agent isn't acting; you are.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import vizdoom as vzd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from doom.campaign import CAMPAIGN_BUTTONS, campaign_actions, default_wad
from rl.bc import nearest_action, save_demo


def main() -> None:
    p = argparse.ArgumentParser(description="Record human SPECTATOR demos.")
    p.add_argument("--map", default="MAP01")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default=None, help="Demos dir (default: <memory>/demos).")
    p.add_argument("--strafe", action="store_true", help="Match a strafe-action training set.")
    p.add_argument("--minutes", type=float, default=10.0,
                   help="Max minutes per episode for the HUMAN to reach the exit (default 10). "
                        "The training timeout is short for PPO efficiency; a human needs more.")
    args = p.parse_args()

    cfg = Config()
    out_dir = args.out or os.path.join(cfg.memory_dir, "demos")
    actions = [vec for vec, _ in _action_vectors(args.strafe)]
    w, h = cfg.resolution

    game = vzd.DoomGame()
    game.set_doom_game_path(cfg.wad_path or default_wad())
    game.set_doom_map(args.map)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)         # <- the human drives
    game.set_available_buttons(CAMPAIGN_BUTTONS)
    # A HUMAN needs minutes to navigate to the exit — NOT the short training timeout (which is
    # tuned for PPO sample efficiency). Doom runs at 35 tics/second.
    game.set_episode_timeout(int(args.minutes * 60 * 35))
    game.init()

    print(f"[record] SPECTATOR on {args.map}. Play to the EXIT. {args.episodes} episode(s).")
    print("[record] Close the game window any time to STOP — recorded episodes are kept.")
    closed = False
    for ep in range(args.episodes):
        if closed:
            break
        try:
            game.new_episode()
            obs_frames, act_idxs = [], []
            while not game.is_episode_finished():
                state = game.get_state()
                if state is not None:
                    frame = cv2.resize(state.screen_buffer, (w, h), interpolation=cv2.INTER_AREA)
                    obs_frames.append(frame[:, :, None])
                game.advance_action()                  # the human's input is applied here
                pressed = game.get_last_action()       # raw button vector the human held
                if state is not None:
                    act_idxs.append(nearest_action(pressed, actions))
        except vzd.ViZDoomUnexpectedExitException:
            # The human closed the window — not an error. Save what this episode captured.
            closed = True
            print("[record] window closed — stopping after this episode.")
        # Did this run REACH THE EXIT? Episode ended NOT dead and BEFORE the (long) timeout =
        # the human found the exit. BC should clone only these — wandering/dying demos teach
        # the agent to wander/die. None when the window was closed mid-episode (unknown).
        reached_exit = None
        if not closed:
            from doom.env import classify_terminal
            term = classify_terminal(game.is_player_dead(),
                                     game.get_episode_time(), game.get_episode_timeout())
            reached_exit = (term == "exit")
        if obs_frames:
            path = os.path.join(out_dir, f"demo_{args.map}_{ep:03d}.npz")
            save_demo(path, np.asarray(obs_frames, dtype=np.uint8),
                      np.asarray(act_idxs, dtype=np.int64), reached_exit=reached_exit)
            tag = ("✓ reached EXIT" if reached_exit else
                   "✗ did NOT reach exit (won't be used with --only-success)"
                   if reached_exit is False else "? unknown (window closed)")
            print(f"[record] episode {ep}: {len(act_idxs)} frames — {tag} -> {path}")
    try:
        game.close()
    except Exception:
        pass
    print(f"[record] done. Clone with: python -m rl.bc --demos {out_dir}")


def _action_vectors(strafe: bool):
    """Build (button-vector, label) for each discrete action, aligned to CAMPAIGN_BUTTONS."""
    bidx = {b.name: i for i, b in enumerate(CAMPAIGN_BUTTONS)}
    out = []
    for combo, label in campaign_actions(strafe):
        vec = [0] * len(CAMPAIGN_BUTTONS)
        for name in combo:
            vec[bidx[name]] = 1
        out.append((vec, label))
    return out


if __name__ == "__main__":
    main()
