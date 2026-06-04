"""Memory policy — turn the SQLite cognitive memory into decisions that feed the agent.

The autonomy loop's `propose_next` only sees the LAST eval (10 episodes). This module lets
it draw on the WHOLE persistent history instead:

  1. death patterns   — where/how the agent keeps dying (uses the episodic context: health,
                        region, nearest_enemy) → a targeted reward delta.
  2. experiment memory — never re-try a change a past A/B already proved doesn't help
                        (regressed / no_effect) → the agent stops repeating its own mistakes.
  3. adoption         — copy every "improved" experiment's winning knobs into LearnedConfig,
                        so a proven gain persists across sessions.

The pure pieces (death_pattern, failed_params, propose_from_memory) take plain lists and are
unit-tested; the db-backed wrappers just fetch and delegate.
"""
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Knobs this policy may move, with hard bounds (its own guardrails).
_BOUNDS = {
    "DAMAGE_TAKEN_PENALTY": (0.0, 0.5),
    "DEATH_PENALTY":        (1.0, 20.0),
    "COVERAGE_REWARD":      (0.0, 4.0),
    "FRONTIER_REWARD":      (0.0, 0.2),
}


def _clamp(key: str, value: float) -> float:
    lo, hi = _BOUNDS[key]
    return round(max(lo, min(hi, value)), 4)


# ---------------------------------------------------------------------------
# Pure analysis
# ---------------------------------------------------------------------------

def death_pattern(events: List[dict]) -> dict:
    """Summarise deaths across the whole memory. Returns counts/fractions + the dominant
    region and enemy, so a proposal can target the REAL failure mode, not a guess."""
    deaths = [e for e in events if e.get("type") == "death"]
    n = len(deaths)
    if n == 0:
        return {"n": 0, "low_hp_fraction": 0.0, "top_region": None, "top_enemy": None}
    low_hp = sum(1 for e in deaths
                 if e.get("health") is not None and float(e["health"]) <= 30)
    regions = Counter(e.get("region") for e in deaths if e.get("region"))
    enemies = Counter(e.get("nearest_enemy") for e in deaths if e.get("nearest_enemy"))
    return {
        "n": n,
        "low_hp_fraction": low_hp / n,
        "top_region": regions.most_common(1)[0][0] if regions else None,
        "top_enemy": enemies.most_common(1)[0][0] if enemies else None,
    }


def failed_params(experiments: List[dict]) -> Dict[str, str]:
    """{KNOB: most-recent verdict} for knobs a past experiment found 'regressed' or
    'no_effect'. The proposer avoids re-touching these. Experiments are newest-first."""
    seen: Dict[str, str] = {}
    for exp in experiments:  # newest first -> first verdict per knob is the latest
        verdict = exp.get("result")
        if verdict not in ("regressed", "no_effect"):
            continue
        try:
            delta = json.loads(exp.get("param") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        for knob in delta:
            seen.setdefault(knob, verdict)
    return seen


def propose_from_memory(
    events: List[dict],
    experiments: List[dict],
    env: Dict[str, str],
    min_deaths: int = 10,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """A reward delta grounded in the FULL history, or (None, None) if memory has no clear
    signal. Targets the dominant death mode but refuses to repeat a change a past experiment
    already proved useless."""
    if len(events) < min_deaths:
        return None, None
    avoid = failed_params(experiments)
    dp = death_pattern(events)
    if dp["n"] < min_deaths:
        return None, None

    # Dominant failure: dying at low HP -> raise the damage-taken penalty (teach caution),
    # unless an experiment already showed that doesn't help.
    if dp["low_hp_fraction"] >= 0.6 and "DAMAGE_TAKEN_PENALTY" not in avoid:
        cur = float(env.get("DAMAGE_TAKEN_PENALTY", 0.1))
        new = dict(env)
        new["DAMAGE_TAKEN_PENALTY"] = str(_clamp("DAMAGE_TAKEN_PENALTY", cur * 1.5 + 0.05))
        enemy = dp["top_enemy"] or "enemies"
        return new, (f"memory: {dp['low_hp_fraction']:.0%} of {dp['n']} deaths at low HP "
                     f"(often near {enemy}) -> raise DAMAGE_TAKEN_PENALTY to "
                     f"{new['DAMAGE_TAKEN_PENALTY']}")

    # If low-HP isn't the issue, deaths are likely from over-engagement -> nudge DEATH_PENALTY
    # up a touch so risky fights cost more (again, only if not already disproven).
    if dp["low_hp_fraction"] < 0.3 and "DEATH_PENALTY" not in avoid:
        cur = float(env.get("DEATH_PENALTY", 5.0))
        new = dict(env)
        new["DEATH_PENALTY"] = str(_clamp("DEATH_PENALTY", cur * 1.2))
        return new, (f"memory: deaths are not low-HP ({dp['low_hp_fraction']:.0%}) — likely "
                     f"reckless fights -> raise DEATH_PENALTY to {new['DEATH_PENALTY']}")

    return None, None


# ---------------------------------------------------------------------------
# DB-backed wrappers (fetch from SQLite, delegate to the pure functions)
# ---------------------------------------------------------------------------

def recall_proposal(memory_dir: str, env: Dict[str, str]
                    ) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """propose_from_memory over the persisted SQLite memory (rebuilt from JSONL first)."""
    from writer import db as _db
    _db.build(memory_dir)
    events = _db.query_events(memory_dir, limit=5000)
    experiments = _db.query_experiments(memory_dir, limit=100)
    return propose_from_memory(events, experiments, env)


def adopt_improved_experiments(memory_dir: str) -> Dict[str, str]:
    """Copy every 'improved' experiment's winning knobs into LearnedConfig. Returns the flat
    {KNOB: value} adopted (so the caller can log it). Idempotent."""
    from writer import db as _db
    from writer.learned_config import LearnedConfig

    experiments = _db.query_experiments(memory_dir, limit=200)
    learned = LearnedConfig(memory_dir)
    adopted: Dict[str, str] = {}
    # Oldest-first so a newer proof supersedes an older one.
    for exp in reversed(experiments):
        if exp.get("result") != "improved":
            continue
        try:
            delta = json.loads(exp.get("param") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        knobs = {k: v for k, v in delta.items() if k in _BOUNDS or k.isupper()}
        if knobs:
            learned.adopt(knobs, source=f"experiment H{exp.get('hypothesis_id')}",
                          verdict="improved", confidence=float(exp.get("confidence") or 0.0))
            adopted.update({k: str(v) for k, v in knobs.items()})
    return adopted
