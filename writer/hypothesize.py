"""Hypothesis Engine (Phase 2) — behavior flags → falsifiable hypotheses.

Each hypothesis names a metric, predicts a direction, proposes a config change to test,
and carries a confidence score (frequency × recurrence × cross-run consistency).
Hypotheses are stored in the SQLite DB and written to `70-hypotheses/` in the vault.

    python -m writer.hypothesize          # generate hypotheses from current vault
    python -m writer.hypothesize --json   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from writer.behavior import BehaviorFlag


# ---------------------------------------------------------------------------
# Hypothesis dataclass
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    title: str
    body: str
    metric: str         # the variable that should change (e.g. map_explored)
    direction: str      # 'up' or 'down'
    confidence: float   # 0–1
    config_delta: dict  # {ENV_VAR: suggested_value} to apply for the experiment
    source_flag: str    # which BehaviorFlag triggered this


# ---------------------------------------------------------------------------
# Flag → Hypothesis rules
# ---------------------------------------------------------------------------

_RULES = {
    "shoot_spam": Hypothesis(
        title="Raising MISS_PENALTY will reduce shoot-spam and improve accuracy",
        body=(
            "The agent fires continuously with near-zero accuracy — spray-and-pray. "
            "A higher miss penalty should make random firing costly enough that the "
            "agent learns to aim before shooting. Predicted: accuracy > 10%."
        ),
        metric="shooting_accuracy",
        direction="up",
        confidence=0.0,   # filled at runtime
        config_delta={"MISS_PENALTY": "0.15"},
        source_flag="shoot_spam",
    ),
    "low_exploration": Hypothesis(
        title="Raising FRONTIER_REWARD will push exploration past the 20% plateau",
        body=(
            "Coverage stagnates below 20%. FRONTIER_REWARD only pays for net outward "
            "progress — spinning in circles can't farm it. Raising it should drive the "
            "agent to explore new areas instead of replaying known corridors. "
            "Predicted: mean map_explored increases from current value by ≥ 10pp."
        ),
        metric="map_explored",
        direction="up",
        confidence=0.0,
        config_delta={"FRONTIER_REWARD": "0.03", "MOVE_REWARD": "0.0001"},
        source_flag="low_exploration",
    ),
    "passive": Hypothesis(
        title="Raising KILL_REWARD will overcome passive avoidance of enemies",
        body=(
            "Agent achieves <0.5 kills/episode despite enemies being present. "
            "The kill signal is not strong enough relative to movement rewards. "
            "Raising KILL_REWARD should make engaging enemies the dominant strategy. "
            "Predicted: kills/episode > 2."
        ),
        metric="kills_per_episode",
        direction="up",
        confidence=0.0,
        config_delta={"KILL_REWARD": "8.0"},
        source_flag="passive",
    ),
    "circling": Hypothesis(
        title="Enabling FRONTIER_REWARD=0.02 will break the circling behavior",
        body=(
            "The agent's coverage growth rate is near zero across checkpoints, "
            "indicating circling or route-repetition. FRONTIER_REWARD rewards only "
            "NEW maximum distance from spawn — a circling trajectory cannot farm it. "
            "Predicted: coverage growth rate > 5% across 5 checkpoints."
        ),
        metric="map_explored",
        direction="up",
        confidence=0.0,
        config_delta={"FRONTIER_REWARD": "0.02", "MOVE_REWARD": "0.0001",
                      "COVERAGE_REWARD": "1.5"},
        source_flag="circling",
    ),
    "route_repetition": Hypothesis(
        title="Increasing COVERAGE_CELL will break the coverage plateau",
        body=(
            "Coverage is flat across episodes on the same map — the agent revisits "
            "the same small area. Coarser grid cells mean the starting-room rewards "
            "dry up faster, forcing the agent to push further. "
            "Predicted: coverage variance increases across episodes."
        ),
        metric="map_explored",
        direction="up",
        confidence=0.0,
        config_delta={"COVERAGE_CELL": "192.0"},
        source_flag="route_repetition",
    ),
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(flags: List[BehaviorFlag]) -> List[Hypothesis]:
    """Turn a list of BehaviorFlags into falsifiable hypotheses."""
    hypotheses: List[Hypothesis] = []
    for flag in flags:
        template = _RULES.get(flag.name)
        if template is None:
            continue
        import dataclasses
        h = dataclasses.replace(template, confidence=round(flag.confidence, 2))
        hypotheses.append(h)
    return hypotheses


def generate_from_vault(cfg) -> List[Hypothesis]:
    """Detect flags from the vault and generate hypotheses."""
    from writer.behavior import detect_from_vault
    flags = detect_from_vault(cfg)
    return generate(flags)


# ---------------------------------------------------------------------------
# Persist + vault note
# ---------------------------------------------------------------------------

def save_hypotheses(cfg, hypotheses: List[Hypothesis]) -> List[int]:
    """Insert hypotheses into the SQLite DB. Returns list of row ids."""
    from writer import db as _db
    ids = []
    for h in hypotheses:
        row_id = _db.insert_hypothesis(
            cfg.memory_dir,
            title=h.title,
            body=h.body,
            metric=h.metric,
            direction=h.direction,
            confidence=h.confidence,
        )
        ids.append(row_id)
    return ids


def write_hypotheses_note(cfg, hypotheses: List[Hypothesis]) -> str:
    """Write `70-hypotheses/Hypotheses.md` to the vault. Returns path."""
    out_dir = os.path.join(cfg.vault_path, "70-hypotheses")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Hypotheses.md")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "---",
        "type: hypotheses",
        f"created: {ts}",
        f"count: {len(hypotheses)}",
        "tags: [hypotheses, cognitive-loop, doom-rl]",
        "---",
        "",
        "# Active Hypotheses",
        "",
        "_Auto-generated from behavioral flags. Each hypothesis names a metric and a "
        "predicted direction — it is falsifiable. Results link back here._",
        "",
        "| # | Hypothesis | Metric | Direction | Confidence | Config delta |",
        "|---|-----------|--------|-----------|------------|--------------|",
    ]
    for i, h in enumerate(hypotheses, 1):
        delta_str = " ".join(f"`{k}={v}`" for k, v in h.config_delta.items())
        lines.append(
            f"| {i} | {h.title} | `{h.metric}` | {h.direction} "
            f"| {h.confidence:.0%} | {delta_str} |"
        )

    lines += ["", "---", ""]
    for i, h in enumerate(hypotheses, 1):
        lines += [
            f"## H{i}: {h.title}",
            "",
            h.body,
            "",
            f"- **Metric:** `{h.metric}`  **Direction:** {h.direction}",
            f"- **Confidence:** {h.confidence:.0%}  (from `{h.source_flag}` flag)",
            f"- **Config delta:** "
            + "  ".join(f"`{k}={v}`" for k, v in h.config_delta.items()),
            f"- **Status:** open — needs [[Experiment]]",
            "",
            "> To test: apply the config delta, run for ≥ 200k steps, "
            "compare deterministic eval on `" + h.metric + "`.",
            "",
            "---",
            "",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate falsifiable hypotheses from behavioral flags."
    )
    p.add_argument("--json", action="store_true", help="Output as JSON.")
    args = p.parse_args()

    from config import Config
    cfg = Config()
    hypotheses = generate_from_vault(cfg)

    if not hypotheses:
        print("[hypothesize] No flags → no hypotheses. Run `doom-cli behavior` first.")
        return

    if args.json:
        import dataclasses
        print(json.dumps([dataclasses.asdict(h) for h in hypotheses], indent=2))
        return

    for h in hypotheses:
        print(f"[H conf={h.confidence:.0%}] {h.title}")
        print(f"   metric={h.metric} direction={h.direction}")
        print(f"   delta={h.config_delta}")
        print()

    save_hypotheses(cfg, hypotheses)
    path = write_hypotheses_note(cfg, hypotheses)
    print(f"[hypothesize] wrote {path}")


if __name__ == "__main__":
    main()
