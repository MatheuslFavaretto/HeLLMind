"""Phase 2: the knowledge-graph hub (MOC) connects runs/maps/concepts/lessons."""
import os

from config import Config
from writer.note_writer import NoteWriter


def _writer(tmp_path):
    cfg = Config()
    cfg.vault_path = str(tmp_path)
    cfg.memory_dir = os.path.join(str(tmp_path), ".memory")
    cfg.run_name = "run-x"
    cfg.ollama_host = "http://localhost:1"  # never called in this test
    return NoteWriter(cfg, button_names=["ATTACK"])


def test_hub_links_existing_notes(tmp_path):
    w = _writer(tmp_path)
    # seed a few notes across folders
    open(os.path.join(w.dir_maps, "Map - MAP01.md"), "w").close()
    open(os.path.join(w.dir_concept, "Concept - Policy Entropy.md"), "w").close()
    w.registry.register("Policy Entropy", 1)
    os.makedirs(w.dir_lessons, exist_ok=True)
    open(os.path.join(w.dir_lessons, "Lessons.md"), "w").close()

    path = w.write_knowledge_hub()
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert "Knowledge Graph" in text
    assert "[[run-x]]" in text                       # the run note (made in __init__)
    assert "[[Map - MAP01]]" in text                 # maps section
    assert "[[Concept - Policy Entropy]]" in text    # concepts section
    assert "[[Lessons]]" in text                     # lessons linked when present


def test_hub_excludes_synthesis_from_runs(tmp_path):
    w = _writer(tmp_path)
    open(os.path.join(w.dir_runs, "run-x - Synthesis.md"), "w").close()
    text = open(w.write_knowledge_hub(), encoding="utf-8").read()
    # synthesis shows under its own section, not duplicated as a plain run bullet
    assert "[[run-x - Synthesis]]" in text
