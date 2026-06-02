"""Statistics about the distribution of actions chosen by the policy."""
from typing import Dict, List

import numpy as np


def action_distribution(counts: np.ndarray, button_names: List[str]) -> Dict[str, float]:
    """Fraction of each action over the window total."""
    total = counts.sum()
    if total == 0:
        return {name: 0.0 for name in button_names}
    return {button_names[i]: float(counts[i] / total) for i in range(len(button_names))}


def action_entropy(counts: np.ndarray) -> float:
    """Entropy (in nats) of the empirical action distribution.

    High = policy exploring/varied; low = policy collapsing onto one action.
    Useful to detect when the agent has "locked" into a behavior.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def max_entropy(n_actions: int) -> float:
    """Maximum possible entropy (uniform distribution), for normalization."""
    return float(np.log(n_actions)) if n_actions > 1 else 0.0
