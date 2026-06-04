"""Algorithm selection: standard PPO (feed-forward) or RecurrentPPO (LSTM).

An LSTM policy carries hidden state across steps, so the agent can act on temporal
context a single frame doesn't reveal (e.g. "I just came from there"). It's opt-in via
`USE_LSTM` because the saved brain's format differs: an LSTM brain and a feed-forward
brain are NOT interchangeable. The checkpoint name is tagged (`..._lstm`) so a resume
never tries to load one into the other.

`predict()` has the same signature on both classes (the extra `state`/`episode_start`
are simply unused by PPO), so callers can drive either with one recurrent-safe loop.
"""
from typing import Tuple, Type


def algo_class(use_lstm: bool) -> Type:
    """The SB3 algorithm class to use."""
    if use_lstm:
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


def policy_name(use_lstm: bool = False, game_vars: bool = False) -> str:
    """The SB3 policy string. MultiInputPolicy when the obs is a Dict (game_vars)."""
    if game_vars:
        return "MultiInputLstmPolicy" if use_lstm else "MultiInputPolicy"
    return "CnnLstmPolicy" if use_lstm else "CnnPolicy"


def brain_prefix(task: str, num_actions: int, use_lstm: bool,
                 spatial_memory: bool = False, depth_perception: bool = False,
                 automap: bool = False, frame_stack: int = 4, game_vars: bool = False) -> str:
    """The single source of truth for a checkpoint family name. Every site that builds or
    looks up a brain name MUST use this so train/eval/progress/gif never disagree. Any flag
    that changes the obs shape (spatial, depth, automap, frame_stack, game_vars) or the policy
    class (lstm) gets its own tag, so incompatible brains can never share a name and
    cross-load into a crash."""
    return (f"ppo_{task}_a{num_actions}{policy_tag(use_lstm)}"
            f"{spatial_tag(spatial_memory)}{depth_tag(depth_perception)}{automap_tag(automap)}"
            f"{framestack_tag(frame_stack)}{gamevars_tag(game_vars)}")


def describe(use_lstm: bool) -> Tuple[str, str]:
    """(algo name, policy name) for logging."""
    return ("RecurrentPPO" if use_lstm else "PPO"), policy_name(use_lstm)
