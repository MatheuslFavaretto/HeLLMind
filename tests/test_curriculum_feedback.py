"""Closed loop: memory of deaths weights the campaign curriculum (more deaths = more steps)."""
from rl.campaign_callbacks import MapCurriculumCallback, map_step_weights

MAPS = ["MAP01", "MAP02", "MAP03"]


def test_weights_uniform_without_memory():
    w = map_step_weights([], MAPS)
    assert all(abs(v - 1.0) < 1e-9 for v in w.values())  # mean 1.0, all equal


def test_weights_favor_deadly_maps():
    events = (
        [{"type": "death", "map": "MAP02"}] * 9
        + [{"type": "death", "map": "MAP01"}] * 1
        + [{"type": "success", "map": "MAP03"}] * 5  # successes don't add weight
    )
    w = map_step_weights(events, MAPS)
    assert w["MAP02"] > w["MAP01"] > w["MAP03"]      # deadliest gets the most
    assert abs(sum(w.values()) / len(MAPS) - 1.0) < 1e-9  # normalized to mean 1.0


def test_curriculum_budgets_scale_with_weights():
    w = {"MAP01": 0.5, "MAP02": 1.5, "MAP03": 1.0}
    cb = MapCurriculumCallback(MAPS, steps_per_map=100_000, weights=w)
    assert cb.budgets == [50_000, 150_000, 100_000]


def test_curriculum_default_budgets_uniform():
    cb = MapCurriculumCallback(MAPS, steps_per_map=100_000)
    assert cb.budgets == [100_000, 100_000, 100_000]
