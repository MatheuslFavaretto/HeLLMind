"""Persistent cognitive memory (Phase 1).

Stores episode-end events (death/success/timeout, with context) in an append-only
JSONL that PERSISTS across runs, so the system can learn between executions. Writing
an event is a single appended line (microseconds), and it only happens on episode end
— well within the ±2% training-budget rule. Nothing here is read inside the PPO loop;
the LLM "lessons" (Phase 4) consume this offline.

Layout (default: <vault>/.memory/):
    episodic/events.jsonl   ← every relevant event, across runs
    lessons/lessons.jsonl   ← history of extracted lessons
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from writer.snapshot_log import _sanitize


class MemoryStore:
    def __init__(self, memory_dir: str, run_name: str = "") -> None:
        self.memory_dir = memory_dir
        self.run_name = run_name
        self.episodic_path = os.path.join(memory_dir, "episodic", "events.jsonl")
        self.lessons_path = os.path.join(memory_dir, "lessons", "lessons.jsonl")
        os.makedirs(os.path.dirname(self.episodic_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.lessons_path), exist_ok=True)

    # ------------------------------------------------------------------
    def record_event(self, event: Dict[str, Any]) -> None:
        """Append one event (stamped with run + timestamp). Append-only, persistent."""
        stamped = {
            "run": self.run_name,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **event,
        }
        with open(self.episodic_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(stamped), ensure_ascii=False) + "\n")

    def save_lesson_batch(self, lessons: List[dict]) -> None:
        """Append a batch of extracted lessons to the history."""
        with open(self.lessons_path, "a", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(json.dumps(_sanitize(lesson), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    @staticmethod
    def read_events(memory_dir: str) -> List[Dict[str, Any]]:
        path = os.path.join(memory_dir, "episodic", "events.jsonl")
        out: List[Dict[str, Any]] = []
        if not os.path.exists(path):
            return out
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate a partially-written tail line
        return out
