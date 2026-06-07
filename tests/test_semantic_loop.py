"""Semantic memory wired into the auto loop: record (situation→change→outcome), recall by meaning."""
from rl.autonomous import situation_text, semantic_record, semantic_recall


def test_situation_text_describes_behaviour():
    t = situation_text({"explored_fraction": 0.09, "wasted_shot_rate": 0.72, "aim_offset": 0.93,
                        "reward_breakdown": {"explore": 0.82}})
    assert "wasted_shots 72%" in t and "explored 9%" in t and "reward_explore 82%" in t


def test_record_then_recall_a_similar_situation(tmp_path):
    d = str(tmp_path)
    spray = {"explored_fraction": 0.09, "wasted_shot_rate": 0.72, "aim_offset": 0.93,
             "death_rate": 0.3, "reward_breakdown": {"explore": 0.82, "combat": 0.07}}
    semantic_record(d, spray, {"COVERAGE_REWARD": "0.5", "ENGAGEMENT_REWARD": "0.3"},
                    score=0.45, kept=True, reason="cut explore, sharpen aim")
    similar = {"explored_fraction": 0.11, "wasted_shot_rate": 0.68, "aim_offset": 0.88,
               "death_rate": 0.25, "reward_breakdown": {"explore": 0.79, "combat": 0.09}}
    change, note = semantic_recall(d, similar)
    assert change == {"COVERAGE_REWARD": "0.5", "ENGAGEMENT_REWARD": "0.3"}
    assert "semantic recall" in note


def test_recall_ignores_reverted_changes(tmp_path):
    d = str(tmp_path)
    s = {"explored_fraction": 0.5, "wasted_shot_rate": 0.1, "reward_breakdown": {"combat": 0.5}}
    semantic_record(d, s, {"KILL_REWARD": "20"}, score=-0.1, kept=False, reason="regressed")
    change, _ = semantic_recall(d, s)
    assert change is None            # a reverted/negative outcome is not recalled as "what worked"


def test_recall_empty_memory_is_safe(tmp_path):
    assert semantic_recall(str(tmp_path), {"explored_fraction": 0.2}) == (None, "")
