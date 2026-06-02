"""LLM Reflection Engine (Phase 4): turns the persistent memory into LESSONS.

Reads the episodic events accumulated ACROSS runs (writer.memory_store), aggregates
them into a small statistics report, and asks the LLM for reusable lessons (failure
patterns and the like). Fully offline/post-training — zero impact on the PPO loop.

    python -m writer.reflect                 # reflect over the .env vault's memory
    python -m writer.reflect --model qwen2.5:7b
"""
import argparse
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import Config
from writer.memory_store import MemoryStore

LOW_HP = 30.0  # health threshold (just before death) considered "low HP"


def aggregate_events(events: List[dict]) -> Dict:
    """Pure aggregation of episode events into a stats dict for the prompt."""
    deaths = [e for e in events if e.get("type") == "death"]
    successes = [e for e in events if e.get("type") == "success"]
    timeouts = [e for e in events if e.get("type") == "timeout"]

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    n = len(events)
    low_hp = [e for e in deaths if float(e.get("health", 0)) < LOW_HP]
    return {
        "total": n,
        "runs": len({e.get("run", "") for e in events}),
        "deaths": len(deaths),
        "successes": len(successes),
        "timeouts": len(timeouts),
        "death_rate": (len(deaths) / n) if n else 0.0,
        "mean_health_at_death": mean([float(e.get("health", 0)) for e in deaths]),
        "mean_ammo_at_death": mean([float(e.get("ammo", 0)) for e in deaths]),
        "low_hp_death_rate": (len(low_hp) / len(deaths)) if deaths else 0.0,
        "mean_len_death": mean([float(e.get("length", 0)) for e in deaths]),
        "mean_len_success": mean([float(e.get("length", 0)) for e in successes]),
        "deaths_by_map": dict(Counter(e.get("map", "") for e in deaths if e.get("map"))),
    }


def reflect(cfg: Config, model: Optional[str] = None) -> Optional[str]:
    """Generate the lessons note from the vault's memory. Returns the file path."""
    events = MemoryStore.read_events(cfg.memory_dir)
    if len(events) < cfg.min_events_for_lessons:
        print(f"[reflect] only {len(events)} event(s) (< {cfg.min_events_for_lessons}) "
              "— not enough to extract lessons yet.")
        return None

    stats = aggregate_events(events)
    print(f"[reflect] reflecting over {stats['total']} events from "
          f"{stats['runs']} run(s)...")

    from writer.llm_client import LLMWriter

    llm = LLMWriter(
        model=model or cfg.llm_model,
        host=cfg.ollama_host,
        num_ctx=cfg.llm_num_ctx,
        num_predict=cfg.llm_num_predict,
        keep_alive=cfg.llm_keep_alive,
    )
    try:
        note = llm.generate_lessons(stats)
        lessons = [l.model_dump() for l in note.lessons]
    except Exception as e:
        print(f"[reflect] LLM failed ({e}); skipping.")
        return None
    if not lessons:
        print("[reflect] no lessons returned.")
        return None

    MemoryStore(cfg.memory_dir).save_lesson_batch(lessons)

    out_dir = os.path.join(cfg.vault_path, cfg.dir_lessons)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "Lessons.md")
    body = ["---", "type: lessons",
            f"created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"events: {stats['total']}", f"runs: {stats['runs']}",
            "tags:", "  - lessons", "  - doom-rl", "---", "",
            "# Lessons learned (across runs)", "",
            f"From **{stats['total']}** episode events over **{stats['runs']}** run(s): "
            f"{stats['deaths']} deaths ({stats['death_rate']:.0%}), "
            f"{stats['successes']} successes.", ""]
    for i, l in enumerate(lessons, 1):
        body.append(f"## {i}. {l['title']}\n\n{l['insight']}\n\n_Evidence: {l['evidence']}_\n")
    # Link the maps these lessons came from -> connects lessons into the graph.
    maps = [m for m in (stats.get("deaths_by_map", {}) or {}) if m]
    if maps:
        body.append("## Related maps\n")
        body += [f"- [[Map - {m}]]" for m in maps]
        body.append("")
    # Link the vault's main concepts -> lessons stop being an island in the graph.
    try:
        from writer.concept_registry import ConceptRegistry
        from writer.note_writer import _slug_concept

        reg = ConceptRegistry(os.path.join(cfg.vault_path, ".concept_registry.json"))
        concepts = reg.top(5)
        if concepts:
            body.append("## Related concepts\n")
            body += [f"- [[{_slug_concept(c)}]]" for c in concepts]
            body.append("")
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body))
    print(f"[reflect] wrote {len(lessons)} lesson(s) -> {os.path.join(cfg.dir_lessons, 'Lessons.md')}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Extract lessons from the persistent memory.")
    p.add_argument("--model", default=None, help="Ollama model for the lessons (override).")
    args = p.parse_args()
    reflect(Config(), model=args.model)


if __name__ == "__main__":
    main()
