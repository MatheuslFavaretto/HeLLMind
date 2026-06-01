"""Helpers do NoteWriter: slug do conceito e frontmatter YAML."""
from writer.note_writer import _one_line, _slug_concept, _yaml_frontmatter


def test_one_line_strips_dumped_report():
    # bug real: o 3b jogou o fact-sheet inteiro no headline
    dumped = "# Relatório do checkpoint @ 7,500 timesteps\nJanela: 4 episódios\nMapa: MAP01"
    assert _one_line(dumped, 180) == "Relatório do checkpoint @ 7,500 timesteps"
    assert _one_line("> citação markdown", 50) == "citação markdown"
    assert len(_one_line("x" * 300, 90)) == 90


def test_slug_concept_strips_unsafe_chars():
    assert _slug_concept("Reward Shaping") == "Concept - Reward Shaping"
    assert _slug_concept("Exploração: vs/Exploitation!") == "Concept - Exploração vs Exploitation"


def test_yaml_frontmatter_scalars_and_lists():
    fm = _yaml_frontmatter({"type": "checkpoint", "tags": ["a", "b"], "n": 3})
    lines = fm.splitlines()
    assert lines[0] == "---" and lines[-1] == "---"
    assert "type: checkpoint" in lines
    assert "n: 3" in lines
    assert "tags:" in lines and "  - a" in lines and "  - b" in lines
