"""Intelligent Curriculum (Phase 4) — difficulty scoring + forgetting detection.

Extends the existing combined_map_weights (deaths × under-exploration in
campaign_callbacks.py) with:
  - Difficulty score per map: deaths + low-success + low-coverage + low-kills
  - Forgetting detector: kills or coverage on a map dropped ≥ 30% from its best
    window in the episodic memory → Skill Regression Alert

    python -m rl.curriculum          # print difficulty table for this vault
    python -m rl.curriculum --note   # write vault note too
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Regression threshold: metric fell by this fraction from its best window
FORGETTING_THRESHOLD = 0.30

# Window size (events): rolling window to detect a recent drop
_WINDOW = 20


# ---------------------------------------------------------------------------
# Difficulty score
# ---------------------------------------------------------------------------

@dataclass
class MapDifficulty:
    map_name: str
    death_rate: float        # deaths / total_episodes
    timeout_rate: float      # timeouts / total_episodes
    mean_coverage: float     # mean cells visited (raw)
    mean_kills: float        # mean kills / episode
    score: float             # 0–4 composite (higher = harder)
    n_episodes: int


def difficulty_score(
    map_name: str,
    events: List[dict],
    coverage_scale: float = 200.0,  # cells in a "fully explored" map (heuristic)
) -> Optional[MapDifficulty]:
    """Compute a difficulty score for one map from the episodic memory.

    Score = death_rate + timeout_rate + (1 - cov_fraction) + (1 - kill_fraction)
    Each component is 0–1 so score is 0–4 (higher = harder).
    Returns None if no events for this map.
    """
    map_events = [e for e in events if e.get("map") == map_name]
    if not map_events:
        return None

    n = len(map_events)
    deaths   = sum(1 for e in map_events if e.get("type") == "death")
    timeouts = sum(1 for e in map_events if e.get("type") == "timeout")
    kills    = [float(e.get("kills", 0)) for e in map_events if e.get("kills") is not None]
    covs     = [float(e.get("coverage", 0)) for e in map_events if e.get("coverage") is not None]

    mean_kills = sum(kills) / len(kills) if kills else 0.0
    mean_cov   = sum(covs) / len(covs)   if covs  else 0.0

    # Normalise kills and coverage to [0, 1] using soft caps
    kill_fraction = min(1.0, mean_kills / 5.0)        # 5 kills/ep = fully capable
    cov_fraction  = min(1.0, mean_cov / coverage_scale)

    score = (
        (deaths / n)                 # 0–1: more deaths = harder
        + (timeouts / n)             # 0–1: more timeouts = harder
        + (1.0 - cov_fraction)       # 0–1: less explored = harder
        + (1.0 - kill_fraction)      # 0–1: fewer kills = harder
    )

    return MapDifficulty(
        map_name=map_name,
        death_rate=deaths / n,
        timeout_rate=timeouts / n,
        mean_coverage=mean_cov,
        mean_kills=mean_kills,
        score=round(score, 3),
        n_episodes=n,
    )


def difficulty_weights(
    events: List[dict],
    maps: List[str],
    coverage_scale: float = 200.0,
) -> Dict[str, float]:
    """Return per-map training budget weights based on difficulty score.
    Harder maps get more budget. Normalized to mean 1.0 (compatible with
    MapCurriculumCallback). Maps with no data default to 1.0."""
    raw: Dict[str, float] = {}
    for m in maps:
        d = difficulty_score(m, events, coverage_scale)
        raw[m] = (d.score + 0.1) if d else 1.0   # +0.1 so even easy maps get some budget

    mean = sum(raw.values()) / len(maps)
    return {m: round(raw[m] / mean, 4) for m in maps}


# ---------------------------------------------------------------------------
# Forgetting detection
# ---------------------------------------------------------------------------

@dataclass
class ForgettingAlert:
    map_name: str
    metric: str              # 'kills' or 'coverage'
    peak_value: float
    recent_value: float
    drop_fraction: float     # (peak - recent) / peak
    description: str


def detect_forgetting(
    events: List[dict],
    maps: List[str],
    threshold: float = FORGETTING_THRESHOLD,
    window: int = _WINDOW,
) -> List[ForgettingAlert]:
    """Compare recent performance vs peak to detect skill regression per map.

    For each map, computes the peak (best window) and recent (last `window`
    episodes) mean for kills and coverage. If recent < peak × (1 - threshold),
    emits a ForgettingAlert.
    """
    alerts: List[ForgettingAlert] = []

    for map_name in maps:
        map_events = [e for e in events if e.get("map") == map_name]
        if len(map_events) < window * 2:
            continue  # not enough history to detect forgetting

        for metric in ("kills", "coverage"):
            vals = [
                float(e.get(metric, 0))
                for e in map_events
                if e.get(metric) is not None
            ]
            if len(vals) < window * 2:
                continue

            # Peak = max of any rolling window
            windows = [vals[i: i + window] for i in range(len(vals) - window + 1)]
            peak_val = max(sum(w) / len(w) for w in windows)
            if peak_val < 1e-6:
                continue

            # Recent = mean of last `window` values
            recent_val = sum(vals[-window:]) / window
            drop = (peak_val - recent_val) / peak_val

            if drop >= threshold:
                alerts.append(ForgettingAlert(
                    map_name=map_name,
                    metric=metric,
                    peak_value=round(peak_val, 3),
                    recent_value=round(recent_val, 3),
                    drop_fraction=round(drop, 3),
                    description=(
                        f"{map_name}: {metric} dropped from peak {peak_val:.2f} to "
                        f"{recent_val:.2f} (-{drop:.0%}) — possible catastrophic forgetting"
                    ),
                ))

    return alerts


# ---------------------------------------------------------------------------
# Combined weight: difficulty × forgetting boost
# ---------------------------------------------------------------------------

def smart_weights(
    events: List[dict],
    maps: List[str],
    coverage_scale: float = 200.0,
    forgetting_boost: float = 2.0,
) -> Dict[str, float]:
    """Combine difficulty weights with a forgetting boost.

    Maps that are both hard AND showing skill regression get an extra multiplier
    so the curriculum prioritises rehearsal. Normalized to mean 1.0.
    """
    base = difficulty_weights(events, maps, coverage_scale)
    alerts = detect_forgetting(events, maps)
    regressed = {a.map_name for a in alerts}

    raw = {
        m: base[m] * (forgetting_boost if m in regressed else 1.0)
        for m in maps
    }
    mean = sum(raw.values()) / len(maps)
    return {m: round(raw[m] / mean, 4) for m in maps}


# ---------------------------------------------------------------------------
# Vault note
# ---------------------------------------------------------------------------

def write_curriculum_note(cfg, maps: List[str], events: List[dict]) -> str:
    """Write `40-maps/Curriculum.md` with the difficulty table + forgetting alerts."""
    difficulties = [difficulty_score(m, events) for m in maps]
    alerts = detect_forgetting(events, maps)
    weights = smart_weights(events, maps)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "---",
        "type: curriculum",
        f"created: {ts}",
        f"maps: {', '.join(maps)}",
        "tags: [curriculum, difficulty, doom-rl]",
        "---",
        "",
        "# Intelligent Curriculum",
        "",
        "Difficulty score (0–4) = death_rate + timeout_rate + (1−coverage) + (1−kills). "
        "Higher = harder = more training budget.",
        "",
        "| Map | Score | Deaths | Timeouts | Coverage | Kills/ep | Weight | Episodes |",
        "|-----|-------|--------|----------|----------|----------|--------|----------|",
    ]
    for d in difficulties:
        if d is None:
            continue
        w = weights.get(d.map_name, 1.0)
        lines.append(
            f"| {d.map_name} | **{d.score:.2f}** | {d.death_rate:.0%} | "
            f"{d.timeout_rate:.0%} | {d.mean_coverage:.0f} cells | "
            f"{d.mean_kills:.1f} | {w:.2f}× | {d.n_episodes} |"
        )

    if alerts:
        lines += [
            "",
            "## ⚠️ Skill Regression Alerts",
            "",
            "_These maps show significant performance drops vs their historical peak:_",
            "",
        ]
        for a in alerts:
            lines.append(f"- **{a.map_name}** — {a.metric}: "
                         f"{a.peak_value:.2f} → {a.recent_value:.2f} (-{a.drop_fraction:.0%}) "
                         f"→ curriculum weight boosted (rehearsal)")
    else:
        lines += ["", "_No forgetting alerts — performance is stable._"]

    lines += ["", "## Related", ""]
    for m in maps:
        lines.append(f"- [[Map - {m}]]")

    out_dir = os.path.join(cfg.vault_path, cfg.dir_maps)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Curriculum.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Show curriculum difficulty scores and forgetting alerts."
    )
    p.add_argument("--note", action="store_true", help="Also write vault note.")
    args = p.parse_args()

    from config import Config
    from writer.memory_store import MemoryStore
    cfg = Config()
    events = MemoryStore.read_events(cfg.memory_dir)
    maps = list(cfg.maps)

    if not events:
        print("[curriculum] No episodic memory yet — train first.")
        return

    weights = smart_weights(events, maps)
    alerts = detect_forgetting(events, maps)

    print(f"\n{'Map':<8}  {'Score':>5}  {'Deaths':>7}  {'Coverage':>9}  {'Kills/ep':>9}  {'Weight':>7}")
    print("-" * 60)
    for m in maps:
        d = difficulty_score(m, events)
        if d:
            print(f"{m:<8}  {d.score:>5.2f}  {d.death_rate:>7.0%}  "
                  f"{d.mean_coverage:>9.1f}  {d.mean_kills:>9.2f}  {weights[m]:>7.2f}×")
        else:
            print(f"{m:<8}  {'—':>5}  {'—':>7}  {'—':>9}  {'—':>9}  {weights[m]:>7.2f}×")

    if alerts:
        print("\n⚠️  SKILL REGRESSION ALERTS:")
        for a in alerts:
            print(f"  {a.description}")
    else:
        print("\n✅ No forgetting detected.")

    if args.note:
        path = write_curriculum_note(cfg, maps, events)
        print(f"\n[curriculum] wrote {path}")


if __name__ == "__main__":
    main()
