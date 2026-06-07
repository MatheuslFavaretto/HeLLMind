"""A small learned frame embedding for demo retrieval.

The raw-pixel 12x12 descriptor (demo_retrieval.frame_descriptor) only captures coarse brightness
structure, so nearest-neighbour matches are weak. This trains a tiny convolutional autoencoder on
the human demo frames (84x84, single grayscale channel — the same channel we extract from the live
obs) and uses its bottleneck as a SEMANTIC embedding. Demo frames and the live frame pass through
the SAME 1-channel encoder, so they're directly comparable.

Why an autoencoder and not the policy's CNN: the policy expects the full 6-channel stacked obs
(+ game-vars), which a stored 1-channel demo frame can't provide. The AE sidesteps that — it learns
purely from the pixel frames we actually have.

    python -m rl.frame_encoder --train          # train on vault demos -> saves encoder
    # then: rl.eval --recall  (auto-loads the encoder if present)
"""
import argparse
import glob
import os
from typing import Optional

import numpy as np

EMBED_DIM = 64
_DEFAULT_NAME = "frame_encoder.pt"


def _build_modules(torch):
    """Return (encoder, decoder) nn.Modules. Encoder: 84x84x1 -> EMBED_DIM. Decoder mirrors it
    back to 84x84 for the reconstruction loss (only the encoder is used at inference)."""
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 4, stride=2), nn.ReLU(),      # 84 -> 41
                nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),     # 41 -> 19
                nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),     # 19 -> 9
            )
            self.fc = nn.Linear(32 * 9 * 9, EMBED_DIM)

        def forward(self, x):
            h = self.conv(x).flatten(1)
            return self.fc(h)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(EMBED_DIM, 32 * 9 * 9)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 3, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, stride=2), nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, stride=2),
            )

        def forward(self, z):
            import torch.nn.functional as F
            h = self.fc(z).view(-1, 32, 9, 9)
            out = self.deconv(h)
            return F.interpolate(out, size=(84, 84), mode="bilinear", align_corners=False)

    return Encoder(), Decoder()


def _load_demo_frames(demos_dir: str) -> np.ndarray:
    """All successful-demo frames as float32 [N,1,84,84] in [0,1]."""
    frames = []
    for path in sorted(glob.glob(os.path.join(demos_dir, "*.npz"))):
        try:
            d = np.load(path)
        except (OSError, ValueError):
            continue
        if "obs" not in d or len(d["obs"]) == 0:
            continue
        if "reached_exit" in d and not bool(d["reached_exit"]):
            continue
        obs = d["obs"].astype(np.float32) / 255.0          # [N,84,84,1]
        frames.append(np.transpose(obs, (0, 3, 1, 2)))     # -> [N,1,84,84]
    if not frames:
        return np.zeros((0, 1, 84, 84), dtype=np.float32)
    return np.concatenate(frames, axis=0)


def train_frame_encoder(demos_dir: str, out_path: Optional[str] = None,
                        epochs: int = 8, batch_size: int = 256, lr: float = 1e-3) -> str:
    """Train the autoencoder on the demo frames; save the encoder weights. Returns the path."""
    import torch
    frames = _load_demo_frames(demos_dir)
    if len(frames) == 0:
        raise SystemExit(f"No usable demo frames in {demos_dir}")
    out_path = out_path or os.path.join(os.path.dirname(demos_dir.rstrip("/")), _DEFAULT_NAME)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc, dec = _build_modules(torch)
    enc, dec = enc.to(device), dec.to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    data = torch.from_numpy(frames)
    n = len(data)
    print(f"[frame_encoder] training on {n} frames, {epochs} epochs, device={device}")
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            batch = data[perm[i:i + batch_size]].to(device)
            opt.zero_grad()
            recon = dec(enc(batch))
            loss = ((recon - batch) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        print(f"  epoch {ep + 1}/{epochs}  recon_mse={total / n:.5f}")
    torch.save({"embed_dim": EMBED_DIM, "state_dict": enc.state_dict()}, out_path)
    print(f"[frame_encoder] saved encoder -> {out_path}")
    return out_path


class FrameEncoder:
    """Loads a trained encoder and embeds single 84x84 grayscale frames (L2-normalised)."""

    def __init__(self, path: str):
        import torch
        ckpt = torch.load(path, map_location="cpu")
        self._torch = torch
        self.enc, _ = _build_modules(torch)
        self.enc.load_state_dict(ckpt["state_dict"])
        self.enc.eval()
        self.embed_dim = ckpt.get("embed_dim", EMBED_DIM)

    def embed(self, frame: np.ndarray) -> np.ndarray:
        """frame: HxW or HxWx1 grayscale (any 0..255 or 0..1). -> L2-normalised float32 vector."""
        torch = self._torch
        f = np.asarray(frame, dtype=np.float32)
        if f.ndim == 3:
            f = f[..., 0]
        if f.max() > 1.0:
            f = f / 255.0
        t = torch.from_numpy(f).view(1, 1, *f.shape)
        with torch.no_grad():
            v = self.enc(t).cpu().numpy()[0]
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 1e-6 else v.astype(np.float32)


def load_encoder_if_present(memory_dir: str) -> Optional["FrameEncoder"]:
    """Return a FrameEncoder if memory_dir/frame_encoder.pt exists, else None."""
    path = os.path.join(memory_dir, _DEFAULT_NAME)
    if os.path.exists(path):
        try:
            return FrameEncoder(path)
        except Exception as e:                              # corrupt/incompatible checkpoint
            print(f"[frame_encoder] could not load {path}: {e}")
    return None


def main():
    p = argparse.ArgumentParser(description="Train the demo frame-embedding autoencoder.")
    p.add_argument("--demos", default=None, help="Demos dir (default: <vault>/.memory/demos).")
    p.add_argument("--epochs", type=int, default=8)
    args = p.parse_args()
    from config import Config
    cfg = Config()
    demos = args.demos or os.path.join(cfg.memory_dir, "demos")
    train_frame_encoder(demos, epochs=args.epochs)


if __name__ == "__main__":
    main()
