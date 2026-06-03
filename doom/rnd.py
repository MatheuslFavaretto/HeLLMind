"""Random Network Distillation (RND) — intrinsic exploration bonus.

Position-based variant: input is the normalized (x, y) position, not the full
84×84×4 observation (that would be 28k inputs — too heavy for an online predictor).
Spatial RND is exactly what we need: high bonus in unfamiliar map areas, zero in
known spots. Covers the same ground as count-based exploration but never saturates.

Bonus = ||predictor(pos) - target(pos)||² / running_std   (normalized, dimensionless)

The target network is fixed (random init). The predictor is trained online (Adam,
lr=1e-3). Prediction error is high for novel positions → large bonus → agent
is intrinsically motivated to explore new areas.

Usage (campaign.py):
    from doom.rnd import RNDModule
    self._rnd = RNDModule() if cfg.use_rnd else None
    ...
    if self._rnd:
        rnd_bonus = self._rnd.bonus(pos_x, pos_y) * self._rnd_scale
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Pure-numpy version (no extra dependency — SB3 already requires torch,
# but keeping RND in numpy keeps the env lean and the tests fast).
# ---------------------------------------------------------------------------

_FEATURE_DIM = 32   # output dimension of both networks
_HIDDEN_DIM  = 64


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class _Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = (2.0 / in_dim) ** 0.5
        self.W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * scale
        self.b = rng.standard_normal(out_dim).astype(np.float32) * 0.1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b


class _TwoLayerNet:
    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0) -> None:
        self.l1 = _Linear(in_dim, hidden, seed)
        self.l2 = _Linear(hidden, out_dim, seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.l2(_relu(self.l1(x)))


class RNDModule:
    """Spatial RND: bonus for visiting new (x, y) positions.

    Args:
        map_scale: approximate half-width of the map in Doom units. Used to
            normalise coordinates to [-1, 1]. 2048 covers most Doom maps.
        rnd_scale: multiplier on the normalised bonus before it's added to the
            shaped reward. Tune via RND_SCALE in .env (default 0.5).
        lr: Adam learning rate for the predictor network.
    """

    def __init__(
        self,
        map_scale: float = 2048.0,
        rnd_scale: float = 0.5,
        lr: float = 1e-3,
    ) -> None:
        self.map_scale = map_scale
        self.rnd_scale = rnd_scale

        # Fixed random target
        self._target = _TwoLayerNet(2, _HIDDEN_DIM, _FEATURE_DIM, seed=42)
        # Online-trained predictor (same architecture, different init)
        self._pred   = _TwoLayerNet(2, _HIDDEN_DIM, _FEATURE_DIM, seed=99)

        # Adam state for the predictor
        self._lr = lr
        self._m  = [np.zeros_like(p) for p in self._pred_params()]
        self._v  = [np.zeros_like(p) for p in self._pred_params()]
        self._t  = 0

        # Running normalisation of the raw error
        self._mean: float = 0.0
        self._var:  float = 1.0
        self._n:    int   = 0

    # ------------------------------------------------------------------
    def bonus(self, pos_x: float, pos_y: float) -> float:
        """Return the normalised intrinsic bonus for position (pos_x, pos_y)
        and update the predictor with one gradient step."""
        inp = self._encode(pos_x, pos_y)
        target_feat = self._target.forward(inp)
        pred_feat   = self._pred.forward(inp)
        error_vec   = pred_feat - target_feat
        raw_error   = float(np.dot(error_vec, error_vec) / _FEATURE_DIM)

        self._update_predictor(inp, target_feat)
        self._update_stats(raw_error)

        # Normalise: subtract running mean, divide by std
        normalised = (raw_error - self._mean) / max(self._var ** 0.5, 1e-6)
        # Clip to avoid outlier spikes on the first steps
        return float(np.clip(normalised, -3.0, 10.0)) * self.rnd_scale

    # ------------------------------------------------------------------
    def _encode(self, x: float, y: float) -> np.ndarray:
        return np.array([x / self.map_scale, y / self.map_scale], dtype=np.float32)

    def _pred_params(self):
        """Flat list of all predictor weight/bias arrays (for Adam)."""
        return [self._pred.l1.W, self._pred.l1.b,
                self._pred.l2.W, self._pred.l2.b]

    def _update_predictor(self, inp: np.ndarray, target: np.ndarray) -> None:
        """One Adam step on MSE(pred, target)."""
        # Forward (again, for gradients)
        h = _relu(self._pred.l1(inp))
        out = self._pred.l2(h)
        delta2 = 2.0 * (out - target) / _FEATURE_DIM   # dL/d_out

        # Backprop through l2
        dW2 = np.outer(delta2, h)
        db2 = delta2
        # Backprop through relu + l1
        delta1 = (self._pred.l2.W.T @ delta2) * (h > 0).astype(np.float32)
        dW1 = np.outer(delta1, inp)
        db1 = delta1

        grads = [dW1, db1, dW2, db2]
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self._pred_params(), grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            m_hat = self._m[i] / (1 - b1 ** self._t)
            v_hat = self._v[i] / (1 - b2 ** self._t)
            p -= self._lr * m_hat / (np.sqrt(v_hat) + eps)

    def _update_stats(self, x: float) -> None:
        """Welford running mean + variance for normalisation."""
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._var = (self._var * (self._n - 1) + delta * delta2) / max(self._n, 1)
