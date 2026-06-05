"""Single entry-point for all LLM documentation (V2-P0 integration).

V1 had 5 separate modules (note_writer, reflect, process_run, compare_runs, suggest).
This facade re-exports the public surface so callers can use one import:

    from writer.documenter import process_run, aggregate_events, suggest, compare

The underlying modules are kept as implementation details and will be progressively merged
into this file in later V2 phases. Nothing is deleted in P0 — we integrate first, cut after
the merge is proven stable.
"""
from writer.note_writer import NoteWriter  # noqa: F401  (re-export)
from writer.process_run import process_run  # noqa: F401
from writer.compare_runs import summarize as compare_runs_summarize  # noqa: F401
from writer.reflect import aggregate_events  # noqa: F401
from writer.suggest import suggest  # noqa: F401


def document_run(cfg, run_name: str, *, use_llm: bool = True) -> None:
    """High-level convenience: process a completed run (notes + lessons + suggestion).
    This is the single call V2 code uses; V1 code keeps calling the sub-modules directly."""
    if not use_llm or not cfg.docs_enabled:
        return
    try:
        process_run(cfg, run_name=run_name)
    except Exception as exc:
        print(f"[documenter] documentation skipped: {exc}")
