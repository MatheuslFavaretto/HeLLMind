"""Non-parametric imitation: at play time, look up the most similar frame in the human
demos and replay the action the human took there.

This is the honest "search memory for the best action" the project wants. Unlike replaying a
recorded tape by step-index, it matches on what the agent ACTUALLY SEES right now: each demo
frame is reduced to a small visual descriptor, and at every step we find the nearest demo
descriptor (k-NN) and reuse its action — but ONLY when the match is close enough (a confidence
gate), otherwise we defer to the policy. The demos are real successful runs to the exit, so the
retrieved actions are "what worked" in a genuinely similar view.

Technique: this is episodic control / nearest-neighbour imitation (cf. Neural Episodic Control).

    from rl.demo_retrieval import DemoRetriever
    r = DemoRetriever("vault/.memory/demos")
    action, sim = r.retrieve(frame_84x84)        # sim in [-1,1]; action is None if sim < gate
"""
import glob
import os
from typing import Optional, Tuple

import numpy as np

# Visual descriptor: downsample the 84x84 grayscale frame to GRID x GRID and L2-normalise.
# Small enough for fast brute-force k-NN over ~15k frames; coarse enough to match "same view"
# despite pixel noise / small motion. 12x12 = 144 dims.
GRID = 12


def frame_descriptor(frame: np.ndarray) -> np.ndarray:
    """84x84 (or any HxW) grayscale frame → GRID*GRID float32 descriptor of its STRUCTURE.
    Block-mean downsample, then MEAN-CENTRE before L2-normalising so the descriptor encodes
    spatial contrast, not absolute brightness. (Without centring, any near-uniform frame —
    including pure noise, which averages to flat gray — normalises to the all-ones direction
    and spuriously matches everything at cosine ~1.0.) A flat frame → ~zero vector → matches
    nothing, which is correct."""
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim == 3:                     # (H,W,1) → (H,W)
        f = f[..., 0]
    h, w = f.shape
    bh, bw = h // GRID, w // GRID
    if bh < 1 or bw < 1:                # frame smaller than the grid: just flatten
        d = f.flatten()
    else:
        # Average each block → (GRID, GRID). Trim any remainder rows/cols first.
        f = f[: bh * GRID, : bw * GRID]
        d = f.reshape(GRID, bh, GRID, bw).mean(axis=(1, 3)).flatten()
    d = d - d.mean()                    # encode contrast/structure, not brightness
    n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 1e-6 else d.astype(np.float32)


class DemoRetriever:
    """Index of (frame descriptor → action) over all human demos; nearest-neighbour lookup."""

    def __init__(self, demos_dir: str, skip_noop: bool = False, noop_action: int = 0,
                 encoder=None):
        """Load every .npz demo and build the descriptor matrix.

        skip_noop drops frames whose action is `noop_action` (the human idling) so retrieval
        suggests an ACTIVE move rather than freezing — off by default (keep it faithful).
        encoder: an optional FrameEncoder (learned embedding). When given, frames are embedded
        with it instead of the coarse pixel descriptor — better at matching 'same situation'."""
        self.encoder = encoder
        self.actions = np.zeros((0,), dtype=np.int64)
        self.descriptors = np.zeros((0, GRID * GRID), dtype=np.float32)
        self.n_demos = 0
        descs, acts = [], []
        embed = (lambda f: encoder.embed(f)) if encoder is not None else frame_descriptor
        for path in sorted(glob.glob(os.path.join(demos_dir, "*.npz"))):
            try:
                d = np.load(path)
            except (OSError, ValueError):
                continue
            if "obs" not in d or "actions" not in d or len(d["actions"]) == 0:
                continue
            # Only learn from demos that actually reached the exit (if flagged).
            if "reached_exit" in d and not bool(d["reached_exit"]):
                continue
            obs, a = d["obs"], d["actions"].astype(np.int64)
            self.n_demos += 1
            for i in range(len(a)):
                if skip_noop and a[i] == noop_action:
                    continue
                descs.append(embed(obs[i]))
                acts.append(int(a[i]))
        if descs:
            self.descriptors = np.stack(descs).astype(np.float32)
            self.actions = np.asarray(acts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.actions)

    def retrieve(self, frame: np.ndarray, min_similarity: float = 0.90
                 ) -> Tuple[Optional[int], float]:
        """Nearest demo frame to `frame`. Returns (action, cosine_similarity).
        action is None when the best match is below `min_similarity` (no confident memory) —
        the caller should then defer to the policy."""
        if len(self) == 0:
            return None, 0.0
        q = self.encoder.embed(frame) if self.encoder is not None else frame_descriptor(frame)
        sims = self.descriptors @ q                      # cosine sim (both normalised)
        j = int(np.argmax(sims))
        best = float(sims[j])
        if best < min_similarity:
            return None, best
        return int(self.actions[j]), best
