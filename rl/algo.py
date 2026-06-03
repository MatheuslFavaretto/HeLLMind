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


def policy_name(use_lstm: bool) -> str:
    """The policy string for image observations."""
    return "CnnLstmPolicy" if use_lstm else "CnnPolicy"


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


def brain_prefix(task: str, num_actions: int, use_lstm: bool,
                 spatial_memory: bool = False, depth_perception: bool = False) -> str:
    """The single source of truth for a checkpoint family name. Every site that builds or
    looks up a brain name MUST use this so train/eval/progress/gif never disagree. Any flag
    that changes the obs shape (spatial, depth) or the policy class (lstm) gets its own tag,
    so incompatible brains can never share a name and cross-load into a crash."""
    return (f"ppo_{task}_a{num_actions}"
            f"{policy_tag(use_lstm)}{spatial_tag(spatial_memory)}{depth_tag(depth_perception)}")


def describe(use_lstm: bool) -> Tuple[str, str]:
    """(algo name, policy name) for logging."""
    return ("RecurrentPPO" if use_lstm else "PPO"), policy_name(use_lstm)
