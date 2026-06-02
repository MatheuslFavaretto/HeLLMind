"""Loop de feedback: o Obsidian deixa de ser só receptor e passa a CONTROLAR.

Creates a control note at `00-index/control.md` with a simple YAML frontmatter.
Every N steps training re-reads this file and adapts WITHOUT restarting:
- `stop_training: true`  -> ends training cleanly.
- `novelty_threshold`    -> more/less sensitive about writing notes (live).
- `write_every_steps`    -> snapshot collection cadence (live).

We keep a tiny parser (no YAML dependency): the frontmatter is just
`chave: valor` entre `---`, o que basta para um painel de controle.
"""
import os
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback


def _coerce(v: str) -> Any:
    s = v.strip().strip('"').strip("'")
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def read_frontmatter(path: str) -> Dict[str, Any]:
    """Read a note's simple YAML frontmatter. Error-tolerant (returns {})."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    out: Dict[str, Any] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = _coerce(val)
    return out


def ensure_control_note(
    path: str, novelty_threshold: float, write_every_steps: int
) -> None:
    """Create the control note with defaults, if it doesn't exist yet."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = (
        "---\n"
        "type: control\n"
        "stop_training: false\n"
        f"novelty_threshold: {novelty_threshold}\n"
        f"write_every_steps: {write_every_steps}\n"
        "---\n\n"
        "# Training control panel\n\n"
        "Edit the **frontmatter** values above while training runs — it re-reads this\n"
        "file every few thousand steps and adapts without restarting.\n\n"
        "- `stop_training: true` ends training cleanly (saves the model).\n"
        "- `novelty_threshold` controls how different something must be to become a note.\n"
        "- `write_every_steps` controls the collection cadence.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


class ControlCallback(BaseCallback):
    """Periodically re-reads the control note and applies its changes to training."""

    def __init__(
        self,
        control_path: str,
        every_steps: int,
        doc_callback: Optional[BaseCallback] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.control_path = control_path
        self.every_steps = max(1, every_steps)
        self.doc_callback = doc_callback
        self._next_check = 0

    def _on_training_start(self) -> None:
        nt = getattr(self.doc_callback, "novelty_threshold", 0.15)
        we = getattr(self.doc_callback, "write_every_steps", 50000)
        ensure_control_note(self.control_path, nt, we)
        if self.verbose:
            print(f"[control] panel at {self.control_path} (edit it to steer training)")

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_check:
            return True
        self._next_check = self.num_timesteps + self.every_steps

        ctrl = read_frontmatter(self.control_path)
        if not ctrl:
            return True

        if ctrl.get("stop_training") is True:
            print(f"[control] stop_training=true — stopping at {self.num_timesteps} steps.")
            return False  # SB3 stops learn() cleanly

        if self.doc_callback is not None:
            nt = ctrl.get("novelty_threshold")
            if isinstance(nt, (int, float)) and nt != self.doc_callback.novelty_threshold:
                if self.verbose:
                    print(f"[control] novelty_threshold -> {nt}")
                self.doc_callback.novelty_threshold = float(nt)
            we = ctrl.get("write_every_steps")
            if isinstance(we, int) and we > 0 and we != self.doc_callback.write_every_steps:
                if self.verbose:
                    print(f"[control] write_every_steps -> {we}")
                self.doc_callback.write_every_steps = int(we)
        return True
