"""Tests for writer.perception_note — the static 'how the agent sees' vault note."""
import os
from types import SimpleNamespace

from writer import perception_note


def test_write_creates_note(tmp_path):
    cfg = SimpleNamespace(vault_path=str(tmp_path), dir_concepts="20-concepts")
    path = perception_note.write(cfg)
    assert os.path.exists(path)
    assert path.endswith("Concept - Agent Perception.md")


def test_note_content_mentions_perception_basics(tmp_path):
    cfg = SimpleNamespace(vault_path=str(tmp_path), dir_concepts="20-concepts")
    path = perception_note.write(cfg)
    text = open(path, encoding="utf-8").read()
    # It should explain the core perception facts.
    assert "pixel" in text.lower()
    assert "exit" in text.lower()
    assert text.startswith("---")  # has frontmatter


def test_write_is_idempotent(tmp_path):
    cfg = SimpleNamespace(vault_path=str(tmp_path), dir_concepts="20-concepts")
    p1 = perception_note.write(cfg)
    p2 = perception_note.write(cfg)
    assert p1 == p2
    assert os.path.exists(p1)
