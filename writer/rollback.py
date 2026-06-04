"""Structured rollback log (P4) — "never degrade the agent permanently".

Every adjustment the auto loop makes is recorded as a structured record:

    {"iter", "before": {...}, "change": {...}, "after": {...}, "result": {...}, "kept": bool}

`before`/`after` hold only the knobs that CHANGED; `change` maps each to [old, new]; `result`
holds the measured outcome (score + key metrics) and whether it was KEPT or ROLLED BACK. This
is the audit trail that makes every tuning decision reversible and inspectable — separate from
the SQLite experiment registry (which is the queryable view); this is the append-only truth.

Layout: `<memory_dir>/rollback.jsonl`
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


class RollbackLog:
    def __init__(self, memory_dir: str) -> None:
        self.path = os.path.join(memory_dir, "rollback.jsonl")

    def record(self, iteration: int, before: Dict[str, Any], change: Dict[str, list],
               after: Dict[str, Any], result: Dict[str, Any], kept: bool) -> dict:
        """Append one structured rollback record. Returns it."""
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iter": int(iteration),
            "before": before, "change": change, "after": after,
            "result": result, "kept": bool(kept),
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def history(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def reverted(self) -> List[dict]:
        """Just the changes that REGRESSED and were rolled back (the safety net firing)."""
        return [r for r in self.history() if not r.get("kept", True)]


def diff_envs(before_env: Dict[str, Any], after_env: Dict[str, Any]):
    """Compute (before, change, after) over only the knobs that changed (string-compared)."""
    change: Dict[str, list] = {}
    before: Dict[str, Any] = {}
    after: Dict[str, Any] = {}
    for k in after_env:
        o, n = before_env.get(k), after_env.get(k)
        if str(o) != str(n):
            change[k] = [o, n]
            before[k] = o
            after[k] = n
    return before, change, after
