"""Behavioral Reflection (Phase 2.5) — detect bad behavior patterns from telemetry.

Reads episodic events and snapshot logs to flag first-class behaviors:
  - shoot_spam   : high fire rate, near-zero accuracy (wastes ammo, indicates fixation)
  - circling     : low net displacement vs distance travelled (the spin bug)
  - low_exploration : map coverage stays below the threshold across episodes
  - passive      : few kills + low damage dealt despite enemies present
  - route_repetition: same area visited every episode (coverage plateau per map)

Each flag carries a confidence (0–1) derived from frequency and magnitude.
Flags link back to a recommendation so `doom-cli behavior` is actionable.

    python -m writer.behavior          # detect flags for the .env vault
    python -m writer.behavior --json   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Flag dataclass
# ---------------------------------------------------------------------------

@dataclass
class BehaviorFlag:
    name: str              # shoot_spam | circling | low_exploration | passive | route_repetition
    confidence: float      # 0–1 (frequency × magnitude)
    description: str       # one-line human-readable finding
    evidence: str          # numbers behind the flag
    recommendation: str    # what to change in .env


# ---------------------------------------------------------------------------
# Thresholds (tunable without changing logic)
# ---------------------------------------------------------------------------

_SHOOT_SPAM_ACC_MAX   = 0.08   # accuracy below this while shooting = spam
_SHOOT_SPAM_FIRE_MIN  = 0.3    # fire-rate fraction of steps that is "a lot"
_CIRCLING_DISP_RATIO  = 0.25   # net_displacement / total_distance < this = circling
_EXPLORATION_LOW      = 0.20   # map coverage fraction below this = low exploration
_PASSIVE_KILLS_MAX    = 0.5    # mean kills/ep below this = passive
_PASSIVE_DAMAGE_MAX   = 5.0    # mean damage dealt below this (when enemies present)
_ROUTE_REP_PLATEAU    = 0.05   # coverage change < 5% across last N checkpoints = plateau


# ---------------------------------------------------------------------------
# Detectors — pure functions over aggregated data
# ---------------------------------------------------------------------------

def detect_shoot_spam(
    snapshots: List[Dict[str, Any]],
) -> Optional[BehaviorFlag]:
    """High shot rate + low accuracy = spray-and-pray (wastes ammo, no reward)."""
    if not snapshots:
        return None
    acc_vals = [float(s.get("shooting_accuracy", 0.0)) for s in snapshots
                if s.get("shooting_accuracy") is not None]
    # fire_rate: shots / steps (approximated from accuracy + hit counts when available)
    if not acc_vals:
        return None
    mean_acc = sum(acc_vals) / len(acc_vals)
    if mean_acc >= _SHOOT_SPAM_ACC_MAX:
        return None
    # Confidence scales with how bad accuracy is (lower = more confident it's spam)
    conf = min(1.0, (_SHOOT_SPAM_ACC_MAX - mean_acc) / _SHOOT_SPAM_ACC_MAX)
    return BehaviorFlag(
        name="shoot_spam",
        confidence=round(conf, 2),
        description=f"Shoot-spam detected: mean accuracy {mean_acc:.1%} (< {_SHOOT_SPAM_ACC_MAX:.0%})",
        evidence=f"mean_accuracy={mean_acc:.3f} over {len(acc_vals)} checkpoint(s)",
        recommendation=(
            "Raise MISS_PENALTY (e.g. 0.05→0.15) to discourage random firing; "
            "or add a short aim-only action that doesn't consume ammo."
        ),
    )


def detect_low_exploration(
    events: List[Dict[str, Any]],
) -> Optional[BehaviorFlag]:
    """Mean coverage per episode below threshold."""
    cov_events = [e for e in events if e.get("coverage") is not None]
    if not cov_events:
        return None
    mean_cov = sum(float(e["coverage"]) for e in cov_events) / len(cov_events)
    # coverage is reported as a fraction (0–1) or cell count; normalise heuristically
    # If > 1 it's a raw cell count; convert to fraction assuming MAP01 has ~1000 cells.
    if mean_cov > 1.0:
        mean_cov = mean_cov / 1000.0
    if mean_cov >= _EXPLORATION_LOW:
        return None
    conf = min(1.0, (_EXPLORATION_LOW - mean_cov) / _EXPLORATION_LOW)
    return BehaviorFlag(
        name="low_exploration",
        confidence=round(conf, 2),
        description=f"Low exploration: mean coverage {mean_cov:.1%}/episode (< {_EXPLORATION_LOW:.0%})",
        evidence=f"mean_coverage={mean_cov:.3f} over {len(cov_events)} episode(s)",
        recommendation=(
            "Raise FRONTIER_REWARD (e.g. 0.01→0.03) to reward directed outward progress. "
            "Ensure MOVE_REWARD > 0 (floor ~0.0001) to prevent passivity. "
            "If coverage stays stuck, consider RND (intrinsic curiosity)."
        ),
    )


def detect_passive(
    events: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
) -> Optional[BehaviorFlag]:
    """Low kills + low damage dealt (agent avoids fighting)."""
    kill_events = [e for e in events if e.get("kills") is not None]
    if not kill_events:
        return None
    mean_kills = sum(int(e["kills"]) for e in kill_events) / len(kill_events)
    if mean_kills >= _PASSIVE_KILLS_MAX:
        return None
    # Also check kills_per_episode from snapshots as a cross-check.
    snap_kills = [float(s.get("kills_per_episode", mean_kills)) for s in snapshots
                  if s.get("kills_per_episode") is not None]
    cross = (sum(snap_kills) / len(snap_kills)) if snap_kills else mean_kills
    combined = (mean_kills + cross) / 2
    conf = min(1.0, (_PASSIVE_KILLS_MAX - combined) / _PASSIVE_KILLS_MAX)
    return BehaviorFlag(
        name="passive",
        confidence=round(conf, 2),
        description=f"Passive behavior: mean {combined:.1f} kills/episode (< {_PASSIVE_KILLS_MAX})",
        evidence=(
            f"mean_kills_events={mean_kills:.2f}  "
            f"mean_kills_snapshots={cross:.2f}  "
            f"episodes={len(kill_events)}"
        ),
        recommendation=(
            "Raise KILL_REWARD (e.g. 5→8) or lower DEATH_PENALTY to reduce cowardice. "
            "Check that MOVE_REWARD is not so high that wandering beats fighting."
        ),
    )


def detect_circling(
    events: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
) -> Optional[BehaviorFlag]:
    """Low net displacement vs total distance — the spin bug.

    Uses coverage plateau as a proxy: if coverage is low AND hasn't grown across
    snapshots, the agent is likely revisiting the same cells (circling or stuck).
    """
    # Primary signal: coverage plateau across consecutive snapshots.
    cov_snaps = [float(s.get("map_explored", s.get("coverage", -1))) for s in snapshots
                 if s.get("map_explored") is not None or s.get("coverage") is not None]
    if len(cov_snaps) < 2:
        # Fallback: if events show very low coverage from the start, flag tentatively.
        cov_events = [e for e in events if e.get("coverage") is not None]
        if not cov_events:
            return None
        mean_cov = sum(float(e["coverage"]) for e in cov_events) / len(cov_events)
        if mean_cov > 1.0:
            mean_cov /= 1000.0
        if mean_cov >= _EXPLORATION_LOW:
            return None
        return BehaviorFlag(
            name="circling",
            confidence=0.4,
            description="Possible circling: coverage flat and very low from the start",
            evidence=f"mean_coverage={mean_cov:.3f} (low-confidence, need more checkpoints)",
            recommendation=(
                "Enable FRONTIER_REWARD=0.02 (rewards net outward progress only — "
                "circling can't farm it). Keep MOVE_REWARD > 0 to prevent full passivity."
            ),
        )

    # Compute growth rate: (last - first) / first
    growth = (cov_snaps[-1] - cov_snaps[0]) / max(cov_snaps[0], 1e-6)
    if growth >= _ROUTE_REP_PLATEAU:
        return None
    conf = min(1.0, max(0.0, (_ROUTE_REP_PLATEAU - growth) / _ROUTE_REP_PLATEAU))
    return BehaviorFlag(
        name="circling",
        confidence=round(conf, 2),
        description=(
            f"Circling / route-repetition: coverage grew only {growth:.1%} "
            f"over {len(cov_snaps)} checkpoint(s)"
        ),
        evidence=(
            f"coverage_start={cov_snaps[0]:.3f}  coverage_end={cov_snaps[-1]:.3f}  "
            f"growth={growth:.3f}"
        ),
        recommendation=(
            "Enable FRONTIER_REWARD=0.02 (only pays for new max distance from spawn). "
            "Set MOVE_REWARD=0.0001 (floor prevents passivity without enabling farming)."
        ),
    )


def detect_route_repetition(
    events: List[Dict[str, Any]],
) -> Optional[BehaviorFlag]:
    """Same area visited every episode — coverage plateau per map."""
    from collections import defaultdict
    by_map: Dict[str, List[float]] = defaultdict(list)
    for e in events:
        if e.get("coverage") is not None and e.get("map"):
            v = float(e["coverage"])
            if v > 1.0:
                v /= 1000.0
            by_map[e["map"]].append(v)

    if not by_map:
        return None

    plateaued = []
    for m, covs in by_map.items():
        if len(covs) < 3:
            continue
        variance = max(covs) - min(covs)
        if variance < _ROUTE_REP_PLATEAU * max(covs, key=abs, default=1.0):
            plateaued.append(m)

    if not plateaued:
        return None

    conf = min(1.0, len(plateaued) / max(len(by_map), 1))
    return BehaviorFlag(
        name="route_repetition",
        confidence=round(conf, 2),
        description=f"Coverage plateau on {plateaued}: agent revisits the same area every episode",
        evidence=f"plateaued_maps={plateaued}",
        recommendation=(
            "Increase COVERAGE_CELL (coarser grid → reward covers more area) or "
            "raise EXIT_REWARD to motivate reaching the level end."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level detector
# ---------------------------------------------------------------------------

def detect(
    events: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
) -> List[BehaviorFlag]:
    """Run all detectors and return every flag whose confidence > 0."""
    detectors = [
        detect_shoot_spam(snapshots),
        detect_low_exploration(events),
        detect_passive(events, snapshots),
        detect_circling(events, snapshots),
        detect_route_repetition(events),
    ]
    return [f for f in detectors if f is not None]


def detect_from_vault(cfg) -> List[BehaviorFlag]:
    """Convenience: load data from the vault config and run all detectors."""
    from writer.memory_store import MemoryStore
    from writer.snapshot_log import SnapshotLog, log_path_for

    events = MemoryStore.read_events(cfg.memory_dir)
    snap_path = log_path_for(cfg.pending_dir, cfg.run_name)
    snapshots = SnapshotLog.read_all(snap_path)
    return detect(events, snapshots)


# ---------------------------------------------------------------------------
# Write vault note
# ---------------------------------------------------------------------------

def write_behavior_note(cfg, flags: List[BehaviorFlag]) -> str:
    """Write `80-recommendations/Behavior.md` with the detected flags. Returns path."""
    from datetime import datetime, timezone
    out_dir = os.path.join(cfg.vault_path, "80-recommendations")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Behavior.md")

    lines = [
        "---",
        "type: behavior-analysis",
        f"created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"flags: {len(flags)}",
        "tags: [behavior, diagnostics, doom-rl]",
        "---",
        "",
        "# Behavioral diagnostics",
        "",
        f"_Generated from telemetry — {len(flags)} flag(s) detected._",
        "",
    ]
    if not flags:
        lines.append("> No behavioral issues detected at this time.")
    else:
        for f in sorted(flags, key=lambda x: -x.confidence):
            icon = "🔴" if f.confidence >= 0.7 else ("🟡" if f.confidence >= 0.4 else "🟢")
            lines += [
                f"## {icon} {f.name.replace('_', ' ').title()}  (confidence {f.confidence:.0%})",
                "",
                f"**Finding:** {f.description}",
                "",
                f"**Evidence:** {f.evidence}",
                "",
                f"**Recommendation:** {f.recommendation}",
                "",
                "---",
                "",
            ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Detect behavioral patterns from HeLLMind telemetry."
    )
    p.add_argument("--json", action="store_true", help="Output as JSON.")
    args = p.parse_args()

    from config import Config
    cfg = Config()
    flags = detect_from_vault(cfg)

    if args.json:
        import dataclasses
        print(json.dumps([dataclasses.asdict(f) for f in flags], indent=2))
        return

    if not flags:
        print("[behavior] No behavioral issues detected.")
        return

    for f in sorted(flags, key=lambda x: -x.confidence):
        icon = "CRITICAL" if f.confidence >= 0.7 else ("WARN" if f.confidence >= 0.4 else "INFO")
        print(f"[{icon}] {f.name} (confidence={f.confidence:.0%})")
        print(f"  {f.description}")
        print(f"  Evidence: {f.evidence}")
        print(f"  Fix: {f.recommendation}")
        print()

    path = write_behavior_note(cfg, flags)
    print(f"[behavior] wrote {path}")


if __name__ == "__main__":
    main()
