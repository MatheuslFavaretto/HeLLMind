"""Snapshot collection during training — FAST, no LLM.

Why it exists: calling Ollama inside the PPO loop freezes training. Here we only
serialize the "new" snapshots (already novelty-filtered) to an append-only JSONL +
a `.meta.json` sidecar. The LLM runs LATER, in `writer.process_run`, so the RL loop
never blocks on model I/O.
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


def _sanitize(obj: Any) -> Any:
    """Make the snapshot 100% serializable (numpy types -> native)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def log_path_for(pending_dir: str, run_name: str) -> str:
    return os.path.join(pending_dir, f"{run_name}.jsonl")


def meta_path_for(pending_dir: str, run_name: str) -> str:
    return os.path.join(pending_dir, f"{run_name}.meta.json")


class SnapshotLog:
    """Writes new snapshots to a JSONL. One instance per run (truncates at start)."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(self.path, "w", encoding="utf-8").close()  # start the run empty
        self.count = 0

    def append(self, snapshot: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(snapshot), ensure_ascii=False) + "\n")
        self.count += 1

    @staticmethod
    def read_all(path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.exists(path):
            return out
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


def write_meta(path: str, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(meta), f, ensure_ascii=False, indent=2)


def read_meta(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
