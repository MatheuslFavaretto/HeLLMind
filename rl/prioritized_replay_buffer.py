"""Prioritized Experience Replay (PER) buffer for DQN.

Implements a segment-tree based priority buffer with TD-error weighting,
importance-sample correction, and beta annealing (as in Rainbow DQN).

Reference: Schaul et al., "Prioritized Experience Replay" (ICLR 2016)
"""
import math
from typing import NamedTuple, Tuple

import numpy as np


class SegmentTree:
    """Efficient range max query and weighted sampling via segment tree.
    
    Supports O(log n) insert/update and O(log n) weighted sampling.
    Used for prioritized experience replay to efficiently sample by priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)  # Internal + leaf nodes
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx: int):
        """Propagate value up the tree after an update."""
        parent = (idx - 1) // 2
        if idx != parent:
            self.tree[parent] = max(self.tree[2 * parent + 1], self.tree[2 * parent + 2])
            self._propagate(parent)

    def update(self, idx: int, priority: float):
        """Set priority at leaf index and propagate up."""
        delta = priority - self.tree[self.capacity + idx]
        self.tree[self.capacity + idx] = priority
        self._propagate(self.capacity + idx - 1)
        return delta

    def add(self, idx: int, priority: float, data):
        """Add/update at index."""
        self.data[idx] = data
        self.update(idx, priority)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Sample indices weighted by priority. Returns (indices, priorities)."""
        total = self.tree[1]  # Root = sum of all priorities
        batch_idxs = np.empty(batch_size, dtype=np.int64)
        batch_priorities = np.empty(batch_size)

        for i in range(batch_size):
            p = rng.uniform(0, total)
            # Binary search for the leaf with this cumsum
            idx = 1
            while idx < self.capacity:
                left = 2 * idx + 1
                right = left + 1
                if p < self.tree[left]:
                    idx = left
                else:
                    p -= self.tree[left]
                    idx = right
            leaf_idx = idx - self.capacity
            batch_idxs[i] = leaf_idx
            batch_priorities[i] = self.tree[idx]

        return batch_idxs, batch_priorities


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with importance-sample correction.
    
    Key parameters:
        alpha: How much to prioritize by TD-error (0 = uniform, 1 = full priority)
        beta: Importance-sample correction exponent (anneals 0→1 over training)
    """

    def __init__(
        self,
        buffer_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_growth: float = 1.0 / 100_000,  # Anneal beta to 1.0 over N steps
        eps: float = 1e-6,
    ):
        """
        Args:
            buffer_size: Max capacity.
            alpha: Priority exponent (0=uniform, 1=full TD-error priority).
            beta: Importance-sample correction exponent (0=none, 1=full correction).
            beta_growth: Per-step increment to beta (to reach 1.0 eventually).
            eps: Small constant to avoid zero priorities.
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_max = 1.0
        self.beta_growth = beta_growth
        self.eps = eps

        self.tree = SegmentTree(buffer_size)
        self.current_idx = 0
        self.size = 0
        self.rng = np.random.default_rng()

    def add(self, data, td_error: float):
        """Add experience with TD-error-based priority.
        
        Args:
            data: Any object (typically (obs, action, reward, next_obs, done)).
            td_error: Absolute temporal-difference error (used for priority).
        """
        priority = (abs(td_error) + self.eps) ** self.alpha

        self.tree.add(self.current_idx, priority, data)
        self.current_idx = (self.current_idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch by priority, with importance-weight correction.
        
        Returns:
            batch_data: Array of sampled (obs, action, reward, next_obs, done) tuples.
            is_weights: Importance-sample correction weights (to multiply loss by).
        """
        assert self.size >= batch_size, f"Buffer size {self.size} < batch_size {batch_size}"

        idxs, priorities = self.tree.sample(batch_size, self.rng)
        batch_data = self.tree.data[idxs]

        # Importance-sample correction: downweight high-priority samples.
        # If a sample is P(i) = p_i / sum(p), its weight in the batch changes its effective
        # importance; we correct by reweighting the loss: w_i = (1 / (N * P(i))) ^ beta.
        probs = priorities / self.tree.tree[1]  # Normalize to probabilities
        weights = (1 / (self.size * probs)) ** self.beta
        weights = weights / weights.max()  # Normalize to [0, 1]

        # Anneal beta toward 1.0 (converge to uniform weighting)
        self.beta = min(self.beta_max, self.beta + self.beta_growth)

        return batch_data, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after compute loss / new TD-error.
        
        Typically called after the model is updated:
            batch_data, weights = buffer.sample(batch_size)
            loss = compute_td_loss(batch_data) * weights
            td_errors = compute_td_error(batch_data)
            buffer.update_priorities(indices, td_errors)
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.eps) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.size
