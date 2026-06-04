"""Long-term knowledge tiers (P3): organise what the agent KNOWS by certainty.

  • FACTS      — measured, high-confidence observations (the bestiary, map difficulty).
  • HYPOTHESES — open questions not yet tested (proposed, awaiting an experiment).
  • VALIDATED  — proven by a multi-seed experiment and/or persisted into learned_config.

This is a READ-ONLY view that aggregates the existing stores (bestiary, SQLite hypotheses/
experiments, learned_config) into the three tiers — it introduces no new storage.
"""
from typing import Any, Dict, List

from writer import db
from writer.bestiary import BestiaryStore, display_name, threat_multipliers
from writer.learned_config import LearnedConfig


def knowledge_tiers(memory_dir: str) -> Dict[str, List[dict]]:
    """Return {"facts": [...], "hypotheses": [...], "validated": [...]}.
    Each item is {"text": str, "evidence": str, "source": str}."""
    return {
        "facts": _facts(memory_dir),
        "hypotheses": _open_hypotheses(memory_dir),
        "validated": _validated(memory_dir),
    }


def _facts(memory_dir: str) -> List[dict]:
    """Ground-truth facts the agent measured. The bestiary is the richest source."""
    out: List[dict] = []
    store = BestiaryStore(memory_dir).load() or {}
    threats = threat_multipliers(store)
    # Sort by how dangerous each monster proved (killed the agent most first).
    for actor, s in sorted(store.items(),
                           key=lambda kv: int((kv[1] or {}).get("killed_agent", 0)),
                           reverse=True):
        if not isinstance(s, dict):
            continue
        enc = int(s.get("encounters", 0) or 0)
        if enc < 1:
            continue
        killed_agent = int(s.get("killed_agent", 0) or 0)
        killed = int(s.get("killed", 0) or 0)
        bits = [f"seen across {enc} run-windows"]
        if killed_agent:
            bits.append(f"killed the agent {killed_agent}×")
        if killed:
            bits.append(f"killed by the agent {killed}×")
        if s.get("ranged"):
            bits.append("ranged attacker")
        text = f"{display_name(actor)}: " + ", ".join(bits)
        ev = f"threat ×{threats[actor]:.2f}" if actor in threats else ""
        out.append({"text": text, "evidence": ev, "source": "bestiary"})
    return out


def _open_hypotheses(memory_dir: str) -> List[dict]:
    out = []
    for h in db.query_hypotheses(memory_dir, status="open"):
        out.append({
            "text": h.get("title", ""),
            "evidence": f"metric={h.get('metric')} {h.get('direction')}, "
                        f"confidence={h.get('confidence', 0):.0%}",
            "source": "hypothesis",
        })
    return out


def _validated(memory_dir: str) -> List[dict]:
    """Things PROVEN: confirmed hypotheses, improved experiments, and persisted knobs."""
    out: List[dict] = []
    for h in db.query_hypotheses(memory_dir, status="confirmed"):
        out.append({"text": h.get("title", ""), "evidence": "hypothesis confirmed",
                    "source": "hypothesis"})
    for e in db.query_experiments(memory_dir):
        if str(e.get("result", "")).lower() == "improved":
            out.append({
                "text": f"{e.get('param')}: {e.get('old_val')} → {e.get('new_val')} helps",
                "evidence": f"experiment verdict, confidence={float(e.get('confidence', 0)):.0%}",
                "source": "experiment",
            })
    for knob, rec in (LearnedConfig(memory_dir).load() or {}).items():
        out.append({
            "text": f"{knob} = {rec.get('value')} (kept)",
            "evidence": f"{rec.get('source', '')}, {rec.get('verdict', 'improved')}",
            "source": "learned_config",
        })
    return out
