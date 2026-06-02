"""Action distribution and entropy."""
import math

import numpy as np

from instrumentation.action_stats import (
    action_distribution,
    action_entropy,
    max_entropy,
)

NAMES = ["A", "B", "C"]


def test_distribution_fractions():
    dist = action_distribution(np.array([2.0, 2.0, 0.0]), NAMES)
    assert dist == {"A": 0.5, "B": 0.5, "C": 0.0}


def test_distribution_all_zero():
    assert action_distribution(np.zeros(3), NAMES) == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_entropy_uniform_is_max():
    assert action_entropy(np.array([1.0, 1.0])) == max_entropy(2)
    assert math.isclose(max_entropy(2), math.log(2))


def test_entropy_collapsed_is_zero():
    assert action_entropy(np.array([10.0, 0.0, 0.0])) == 0.0
    assert action_entropy(np.zeros(3)) == 0.0
