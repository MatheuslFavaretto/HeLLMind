"""NoteWriter helpers: concept slug and YAML frontmatter."""
from writer.note_writer import _one_line, _slug_concept, _strip_citations, _yaml_frontmatter


def test_strip_citations_removes_fake_refs_keeps_values():
    assert _strip_citations("Accuracy rose to 18% (Report, p. Accuracy: 18%)") == "Accuracy rose to 18%"
    assert _strip_citations("Entropy fell (2) and reward rose (3).") == "Entropy fell and reward rose."
    assert _strip_citations("Accuracy is now 25% (25%)") == "Accuracy is now 25% (25%)"  # keeps percentages


def test_one_line_strips_dumped_report():
    # real bug: the 3b dumped the whole fact sheet into the headline
    dumped = "# Checkpoint report @ 7,500 timesteps\nWindow: 4 episodes\nMap: MAP01"
    assert _one_line(dumped, 180) == "Checkpoint report @ 7,500 timesteps"
    assert _one_line("> markdown quote", 50) == "markdown quote"
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
