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


def describe(use_lstm: bool) -> Tuple[str, str]:
    """(algo name, policy name) for logging."""
    return ("RecurrentPPO" if use_lstm else "PPO"), policy_name(use_lstm)
