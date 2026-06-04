"""Tests for instrumentation.game_vars — the invariants that keep var indexing consistent."""
from instrumentation.game_vars import LEVELS, MONOTONIC, TRACKED_VARS, VAR_NAMES


def test_tracked_and_names_same_length():
    # state.game_variables is indexed positionally against VAR_NAMES, so they MUST match.
    assert len(TRACKED_VARS) == len(VAR_NAMES)


def test_var_names_unique():
    assert len(VAR_NAMES) == len(set(VAR_NAMES)), "duplicate var name would alias readings"


def test_monotonic_and_levels_are_subsets_of_var_names():
    for n in MONOTONIC:
        assert n in VAR_NAMES, f"MONOTONIC var {n!r} not in VAR_NAMES"
    for n in LEVELS:
        assert n in VAR_NAMES, f"LEVELS var {n!r} not in VAR_NAMES"


def test_monotonic_and_levels_disjoint():
    # A var is either a cumulative counter (delta) or an instantaneous level, never both.
    assert set(MONOTONIC).isdisjoint(set(LEVELS))


def test_core_vars_present():
    for required in ("killcount", "health", "position_x", "position_y", "angle"):
        assert required in VAR_NAMES
