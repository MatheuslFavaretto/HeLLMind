"""StatsTracker: pontaria, caminho/cobertura, armas e agregação de episódios."""
import numpy as np

from instrumentation.stats_tracker import StatsTracker

BUTTONS = ["MOVE_FORWARD", "TURN_LEFT", "ATTACK"]
ATTACK = 2
MOVE = 0


def _feed(tr, info, action):
    tr.update([info], np.array([action]))


def test_shooting_accuracy_and_counts(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    # 4 ataques: 2 acertam, 2 erram
    for hit in (1.0, 0.0, 1.0, 0.0):
        _feed(tr, make_doom_info(ATTACK, hits=hit), ATTACK)
    # 6 movimentos (não contam como tiro)
    for i in range(6):
        _feed(tr, make_doom_info(MOVE, distance=10.0, pos=(i * 100.0, 0.0)), MOVE)

    snap = tr.snapshot(1000)
    assert snap["shots_fired"] == 4
    assert snap["shots_hit"] == 2
    assert snap["shots_missed"] == 2
    assert snap["shooting_accuracy"] == 0.5


def test_distance_and_coverage(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    for i in range(5):
        _feed(tr, make_doom_info(MOVE, distance=10.0, pos=(i * 200.0, 0.0)), MOVE)
    snap = tr.snapshot(500)
    assert snap["distance_traveled"] == 50.0
    assert snap["cells_visited"] >= 2  # posições espalhadas -> várias células
    assert snap["path_cells"]  # não vazio (alimenta o minimapa)


def test_weapons_distribution(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    for _ in range(3):
        _feed(tr, make_doom_info(MOVE, weapon=2), MOVE)
    for _ in range(1):
        _feed(tr, make_doom_info(MOVE, weapon=3), MOVE)
    snap = tr.snapshot(40)
    w = snap["weapons_used"]
    assert w["slot_2"] == 0.75 and w["slot_3"] == 0.25


def test_episode_aggregation_and_success(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    _feed(tr, make_doom_info(ATTACK, kills=2.0), ATTACK)
    _feed(
        tr,
        make_doom_info(ATTACK, kills=1.0, success=True,
                       episode={"r": 12.5, "l": 200}),
        ATTACK,
    )
    snap = tr.snapshot(100)
    assert snap["episodes"] == 1
    assert snap["success_rate"] == 1.0
    assert snap["mean_reward"] == 12.5
    assert snap["kills_per_episode"] == 3.0  # 2+1 kills / 1 episódio


def test_reset_window_clears(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    _feed(tr, make_doom_info(ATTACK, hits=1.0), ATTACK)
    tr.reset_window()
    snap = tr.snapshot(10)
    assert snap["shots_hit"] == 0
    assert snap["cells_visited"] == 0
    assert snap["episodes"] == 0
