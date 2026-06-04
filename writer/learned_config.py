"""Learned config — the reward knobs the agent has PROVEN help, kept across sessions.

This is the missing half of the self-improvement loop. Today an experiment can prove a
reward change works ("improved" verdict) but the win is forgotten — the next run starts
from the same .env. LearnedConfig closes that: every validated improvement is written here
with its provenance, and training/auto overlay it on boot. The agent therefore ACCUMULATES
what it has proven, instead of re-discovering it every time.

Layout (`<memory>/learned_config.json`):
    {"COVERAGE_REWARD": {"value": "2.5", "source": "experiment H3",
                          "verdict": "improved", "confidence": 0.8, "ts": "..."}}
"""
import json
import os
from datetime import datetime, timezone
from typing import Dict


class LearnedConfig:
    def __init__(self, memory_dir: str) -> None:
        self.path = os.path.join(memory_dir, "learned_config.json")

    # ------------------------------------------------------------------
    def load(self) -> Dict[str, dict]:
        """Full record per knob (value + provenance). {} if nothing learned yet."""
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def values(self) -> Dict[str, str]:
        """Just {KNOB: value} — the flat overlay applied to a training env."""
        return {k: rec.get("value") for k, rec in self.load().items()
                if isinstance(rec, dict) and rec.get("value") is not None}

    # ------------------------------------------------------------------
    def adopt(self, knobs: Dict[str, object], source: str, verdict: str = "improved",
              confidence: float = 0.0) -> Dict[str, dict]:
        """Persist validated knobs (last writer wins — a newer proof supersedes an older).
        Returns the full updated record."""
        rec = self.load()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for k, v in knobs.items():
            rec[k] = {"value": str(v), "source": source, "verdict": verdict,
                      "confidence": float(confidence), "ts": ts}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, self.path)
        return rec

    # ------------------------------------------------------------------
    def apply_to_env(self, env: Dict[str, str]) -> Dict[str, str]:
        """Overlay the learned values onto a training env dict (env wins only where nothing
        was learned). Returns a NEW dict; the input is not mutated."""
        out = dict(env)
        for k, v in self.values().items():
            out[k] = str(v)
        return out
