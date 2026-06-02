"""StatsTracker: aim, path/coverage, weapons and episode aggregation."""
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
    # 6 moves (don't count as shots)
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
    assert snap["cells_visited"] >= 2  # spread-out positions -> several cells
    assert snap["path_cells"]  # non-empty (feeds the minimap)


def test_accuracy_uses_attacked_flag_for_combined_actions(make_doom_info):
    # Campaign uses combined actions (several press ATTACK), so accuracy must come from
    # the env-reported `attacked` flag, not a single ATTACK button index.
    tr = StatsTracker(button_names=["FWD", "FWD+ATK", "ATK"])  # labels, not raw buttons
    # 3 attacks (2 hit), 2 non-attack steps -> accuracy 2/3.
    for attacked, hit in [(True, 1.0), (True, 0.0), (True, 1.0), (False, 0.0), (False, 0.0)]:
        info = make_doom_info(0, hits=hit)
        info["doom"]["attacked"] = attacked
        _feed(tr, info, 0)
    snap = tr.snapshot(10)
    assert snap["shots_fired"] == 3        # counted from the flag, not a button index
    assert snap["shots_hit"] == 2
    assert round(snap["shooting_accuracy"], 3) == 0.667


def test_path_polyline_is_ordered_and_dedups_repeats(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    # Two steps on the same cell (should collapse to one), then move to a new cell.
    _feed(tr, make_doom_info(MOVE, pos=(0.0, 0.0)), MOVE)
    _feed(tr, make_doom_info(MOVE, pos=(10.0, 0.0)), MOVE)   # same grid cell as above
    _feed(tr, make_doom_info(MOVE, pos=(200.0, 0.0)), MOVE)  # new cell
    _feed(tr, make_doom_info(MOVE, pos=(200.0, 200.0)), MOVE)  # new cell
    line = tr.snapshot(40)["path_polyline"]
    assert line == [[0, 0], [2, 0], [2, 2]]  # ordered, consecutive repeat collapsed


def test_path_polyline_keeps_last_completed_episode(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    _feed(tr, make_doom_info(MOVE, pos=(0.0, 0.0)), MOVE)
    _feed(tr, make_doom_info(MOVE, pos=(200.0, 0.0),
                             episode={"r": 1.0, "l": 2}), MOVE)  # episode ends here
    _feed(tr, make_doom_info(MOVE, pos=(800.0, 0.0)), MOVE)      # next episode begins
    line = tr.snapshot(30)["path_polyline"]
    assert line == [[0, 0], [2, 0]]  # the COMPLETED episode, not the fresh one


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
    assert snap["kills_per_episode"] == 3.0  # 2+1 kills / 1 episode


def test_base_return_is_collected(make_doom_info):
    # native (unshaped) episode return is tracked separately from the shaped reward
    tr = StatsTracker(button_names=BUTTONS)
    info = make_doom_info(ATTACK, episode={"r": 5.0, "l": 80})
    info["doom"]["base_return"] = 3.0
    _feed(tr, info, ATTACK)
    snap = tr.snapshot(1)
    assert snap["mean_base_reward"] == 3.0
    assert snap["mean_reward"] == 5.0  # shaped, from the Monitor 'episode' r


def test_coverage_static_scenario_suppressed(make_doom_info):
    # stationary agent (constant position) -> "% explored" must not be an artifact
    tr = StatsTracker(button_names=BUTTONS)
    for _ in range(20):
        _feed(tr, make_doom_info(ATTACK, pos=(0.0, 0.0)), ATTACK)
    cov = tr.snapshot(20)["map_coverage"]
    assert cov.get("static") is True
    assert cov["explored_fraction"] == 0.0


def test_reset_window_clears(make_doom_info):
    tr = StatsTracker(button_names=BUTTONS)
    _feed(tr, make_doom_info(ATTACK, hits=1.0), ATTACK)
    tr.reset_window()
    snap = tr.snapshot(10)
    assert snap["shots_hit"] == 0
    assert snap["cells_visited"] == 0
    assert snap["episodes"] == 0
