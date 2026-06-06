"""Algorithm selection: PPO (feed-forward), RecurrentPPO (LSTM), or QR-DQN (off-policy).

LSTM: carries hidden state across steps (temporal context). Tagged `_lstm` in checkpoints.
QR-DQN: off-policy, distributional DQN with replay buffer (V2 Phase 1). Tagged `qrdqn_`
  prefix instead of `ppo_`. Auto-detected from the checkpoint filename by algo_class_from_path.

`predict()` has the same signature on all classes (extra `state`/`episode_start` are
simply unused by PPO/QR-DQN), so callers can drive any of them with one loop.
"""
import os
from typing import Tuple, Type


def algo_class(use_lstm: bool) -> Type:
    """PPO or RecurrentPPO — for checkpoints that were trained with rl.train."""
    if use_lstm:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    from stable_baselines3 import PPO
    return PPO


def algo_class_from_path(path: str) -> Type:
    """Auto-detect the correct SB3 class from the checkpoint filename.

    qrdqn_* → QRDQN (sb3_contrib)
    *_lstm*  → RecurrentPPO (sb3_contrib)
    else     → PPO (stable_baselines3)

    Use this in eval / autonomous so they work on any checkpoint without a manual flag.
    """
    name = os.path.basename(path).lower()
    if name.startswith("qrdqn"):
        from sb3_contrib import QRDQN
        return QRDQN
    if "_lstm" in name:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    from stable_baselines3 import PPO
    return PPO


def policy_tag(use_lstm: bool) -> str:
    """Checkpoint-name suffix that keeps LSTM and feed-forward brains from cross-loading."""
    return "_lstm" if use_lstm else ""


def spatial_tag(spatial_memory: bool) -> str:
    """Checkpoint-name suffix for spatial-memory brains. Without it, a 2-channel (spatial)
    brain and a 1-channel brain save under the SAME name and cross-load → a hard obs-shape
    crash on eval/resume. Tagging keeps the two families apart, exactly like `_lstm`."""
    return "_sp" if spatial_memory else ""


def depth_tag(depth_perception: bool) -> str:
    """Checkpoint-name suffix for depth-perception brains (extra obs channel → new shape).
    Same cross-load guard as `_sp`/`_lstm`."""
    return "_dp" if depth_perception else ""


def automap_tag(automap: bool) -> str:
    """Checkpoint-name suffix for automap brains (extra obs channel → new shape)."""
    return "_am" if automap else ""


def framestack_tag(frame_stack: int) -> str:
    """Checkpoint-name suffix when frame_stack ≠ the default 4. frame_stack multiplies the
    channel count (the policy input shape), so a 2-stack brain and a 4-stack brain are
    incompatible — same cross-load crash as spatial/depth if they share a name. Default 4
    stays untagged so existing brain names don't change."""
    return f"_fs{frame_stack}" if frame_stack != 4 else ""


def gamevars_tag(game_vars: bool) -> str:
    """Checkpoint-name suffix for game-vars brains. The obs becomes a Dict (image + vector)
    and the policy a MultiInputPolicy — incompatible with a plain CnnPolicy brain."""
    return "_gv" if game_vars else ""


def semantic_tag(semantic_channel: bool) -> str:
    """Checkpoint-name suffix for semantic-channel brains (extra obs channel of detections →
    new input shape). Same cross-load guard as `_sp`/`_dp`."""
    return "_se" if semantic_channel else ""


def policy_name(use_lstm: bool = False, game_vars: bool = False) -> str:
    """The SB3 policy string. MultiInputPolicy when the obs is a Dict (game_vars)."""
    if game_vars:
        return "MultiInputLstmPolicy" if use_lstm else "MultiInputPolicy"
    return "CnnLstmPolicy" if use_lstm else "CnnPolicy"


def brain_prefix(task: str, num_actions: int, use_lstm: bool,
                 spatial_memory: bool = False, depth_perception: bool = False,
                 automap: bool = False, frame_stack: int = 4, game_vars: bool = False,
                 semantic_channel: bool = False) -> str:
    """The single source of truth for a checkpoint family name. Every site that builds or
    looks up a brain name MUST use this so train/eval/progress/gif never disagree. Any flag
    that changes the obs shape (spatial, depth, automap, frame_stack, game_vars, semantic) or
    the policy class (lstm) gets its own tag, so incompatible brains can never share a name and
    cross-load into a crash."""
    return (f"ppo_{task}_a{num_actions}{policy_tag(use_lstm)}"
            f"{spatial_tag(spatial_memory)}{depth_tag(depth_perception)}{automap_tag(automap)}"
            f"{framestack_tag(frame_stack)}{gamevars_tag(game_vars)}"
            f"{semantic_tag(semantic_channel)}")


def describe(use_lstm: bool) -> Tuple[str, str]:
    """(algo name, policy name) for logging."""
    return ("RecurrentPPO" if use_lstm else "PPO"), policy_name(use_lstm)
