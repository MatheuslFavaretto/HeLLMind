"""Post-processing: read the snapshots collected during training and write the notes.

Runs AFTER training (rl.train calls this automatically at the end), so Ollama never
blocks the PPO loop. You can also run it by hand:

    python -m writer.process_run                 # process the .env run (RUN_NAME)
    python -m writer.process_run --run NAME       # a specific run
    python -m writer.process_run --model qwen2.5:7b   # use a better model for notes

Since the LLM now runs in batch, it's worth using a BIGGER model here (e.g. qwen2.5:7b):
training is over, so there's no speed cost — just better notes.
"""
import argparse
import os
from typing import List, Optional

from config import Config
from writer.note_writer import NoteWriter
from writer.snapshot_log import (
    SnapshotLog,
    log_path_for,
    meta_path_for,
    read_meta,
)


def process_run(
    cfg: Config,
    button_names: List[str],
    log_path: Optional[str] = None,
) -> int:
    """Generate notes for all of the run's snapshots. Returns how many were written."""
    log_path = log_path or log_path_for(cfg.pending_dir, cfg.run_name)
    snaps = SnapshotLog.read_all(log_path)
    if not snaps:
        print(f"[process_run] no snapshots at {log_path} — nothing to generate.")
        return 0

    print(
        f"[process_run] {len(snaps)} snapshot(s) | model: {cfg.llm_model} | "
        f"vault: {cfg.vault_path}\n[process_run] generating notes (may take a while)..."
    )
    writer = NoteWriter(cfg, button_names=button_names)
    previous = None
    written = 0
    for i, snap in enumerate(snaps, 1):
        try:
            stem = writer.write_checkpoint(snap, previous=previous)
            written += 1
            print(f"[process_run] {i}/{len(snaps)} -> {stem}")
        except Exception as e:  # one bad note doesn't kill the rest
            print(
                f"[process_run] {i}/{len(snaps)} FAILED "
                f"(step={snap.get('num_timesteps')}): {e}"
            )
        previous = snap

    # Learning curve for the whole run (embedded in the run note).
    try:
        chart = writer.write_run_chart(snaps)
        if chart:
            print(f"[process_run] learning curve: attachments/{chart}")
    except Exception as e:
        print(f"[process_run] curve failed (ignoring): {e}")

    # (A) Narrative synthesis of the whole run.
    try:
        story = writer.write_run_story(snaps)
        if story:
            print(f"[process_run] run synthesis: {story}")
    except Exception as e:
        print(f"[process_run] synthesis failed (ignoring): {e}")

    # (Phase 4) Reflect over the persistent memory to extract cross-run lessons.
    if cfg.memory_enabled:
        try:
            from writer.reflect import reflect

            reflect(cfg)
        except Exception as e:
            print(f"[process_run] reflection failed (ignoring): {e}")

    # (Phase 6) Reward-weight suggestions for human approval (never auto-applied).
    if cfg.memory_enabled and cfg.suggest_rewards:
        try:
            from writer.suggest import suggest

            suggest(cfg)
        except Exception as e:
            print(f"[process_run] suggestions failed (ignoring): {e}")

    # (Phase 2) Knowledge-graph hub connecting runs/maps/concepts/lessons.
    try:
        hub = writer.write_knowledge_hub()
        print(f"[process_run] knowledge hub: {os.path.relpath(hub, cfg.vault_path)}")
    except Exception as e:
        print(f"[process_run] hub failed (ignoring): {e}")

    print(f"[process_run] done: {written}/{len(snaps)} notes in {cfg.vault_path}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the Obsidian notes from snapshots.")
    p.add_argument("--run", default=None, help="Run name (default: RUN_NAME from .env).")
    p.add_argument("--model", default=None, help="Ollama model for the notes (override).")
    args = p.parse_args()

    cfg = Config()
    if args.run:
        cfg.run_name = args.run
    if args.model:
        cfg.llm_model = args.model

    meta = read_meta(meta_path_for(cfg.pending_dir, cfg.run_name)) or {}
    button_names = meta.get("button_names", [])
    if meta.get("scenario"):
        cfg.scenario = meta["scenario"]
    cfg.campaign = bool(meta.get("campaign", cfg.campaign))  # correct note labeling
    if meta.get("maps"):
        cfg.maps = tuple(meta["maps"])
    process_run(cfg, button_names)


if __name__ == "__main__":
    main()
