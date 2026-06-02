"""Regression analysis between checkpoints (feature D).

Makes the documentation INTERPRET, not just describe: when a key metric drops sharply
from one checkpoint to the next, it often means the agent "forgot" something it already
knew (Catastrophic Forgetting). This pure function detects those drops so the NoteWriter
can highlight them and link the concept automatically.
"""
from typing import Dict, List, Optional

# Concept that gets created/linked when a regression is found.
FORGETTING_CONCEPT = "Catastrophic Forgetting"
FORGETTING_DESCRIPTION = (
    "When a network loses skills it already learned while optimizing for something new "
    "— in RL, the agent regresses on metrics it previously mastered."
)

# (snapshot key, readable label, is it a percentage?)
_WATCH = [
    ("shooting_accuracy", "shooting accuracy", True),
    ("mean_reward", "mean reward", False),
    ("kills_per_episode", "kills/episode", False),
    ("success_rate", "success rate", True),
]

# Relative drop (vs. the previous checkpoint) at which we flag a regression.
REGRESSION_DROP = 0.30


def detect_regressions(
    current: Dict, previous: Optional[Dict], threshold: float = REGRESSION_DROP
) -> List[str]:
    """Return descriptions of metrics that dropped >= `threshold` (empty if none)."""
    if not previous:
        return []
    out: List[str] = []
    for key, label, pct in _WATCH:
        cur = float(current.get(key, 0.0))
        prev = float(previous.get(key, 0.0))
        if prev <= 1e-6:  # no positive baseline to call it a "drop"
            continue
        drop = (prev - cur) / prev
        if drop >= threshold:
            cf = f"{cur:.0%}" if pct else f"{cur:,.2f}"
            pf = f"{prev:.0%}" if pct else f"{prev:,.2f}"
            out.append(f"{label} dropped from {pf} to {cf} (-{drop:.0%})")
    return out
