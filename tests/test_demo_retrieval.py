"""Tests for rl.demo_retrieval — nearest-neighbour imitation from human demos.

Guards the two things that actually matter: (1) a frame retrieves the action taken in the most
similar demo frame, and (2) an UNlike frame (noise/flat) does NOT spuriously match — the bug
that the mean-centring fix addresses (without it, any near-uniform frame matched at cosine ~1).
"""
import numpy as np
import pytest

from rl.demo_retrieval import DemoRetriever, frame_descriptor, GRID


def _write_demo(path, obs, actions, reached_exit=True):
    np.savez_compressed(path, obs=obs.astype(np.uint8),
                        actions=np.asarray(actions, dtype=np.int64),
                        reached_exit=np.array(bool(reached_exit)))


def test_descriptor_is_unit_norm_and_centred():
    f = np.random.randint(0, 255, (84, 84, 1), dtype=np.uint8)
    d = frame_descriptor(f)
    assert d.shape == (GRID * GRID,)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)
    assert np.isclose(d.mean(), 0.0, atol=1e-6)          # mean-centred (structure, not brightness)


def test_flat_frame_descriptor_is_zero():
    # A uniform frame has no structure → zero descriptor → matches nothing.
    d = frame_descriptor(np.full((84, 84, 1), 127, dtype=np.uint8))
    assert np.linalg.norm(d) == 0.0


def test_retrieves_action_of_matching_frame(tmp_path):
    # Two distinct structured frames with distinct actions.
    f1 = np.zeros((84, 84, 1), dtype=np.uint8); f1[:42] = 200   # top bright
    f2 = np.zeros((84, 84, 1), dtype=np.uint8); f2[:, :42] = 200  # left bright
    obs = np.stack([f1, f2])
    _write_demo(tmp_path / "d.npz", obs, [3, 7])
    r = DemoRetriever(str(tmp_path))
    assert len(r) == 2
    assert r.retrieve(f1, 0.9)[0] == 3                   # top-bright → its action
    assert r.retrieve(f2, 0.9)[0] == 7                   # left-bright → its action


def test_noise_does_not_match(tmp_path):
    f1 = np.zeros((84, 84, 1), dtype=np.uint8); f1[:42] = 200
    _write_demo(tmp_path / "d.npz", np.stack([f1]), [3])
    r = DemoRetriever(str(tmp_path))
    np.random.seed(0)
    noise = np.random.randint(0, 255, (84, 84, 1), dtype=np.uint8)
    action, sim = r.retrieve(noise, min_similarity=0.9)
    assert action is None                                # below the confidence gate
    assert sim < 0.9


def test_skips_unsuccessful_demos(tmp_path):
    f = np.zeros((84, 84, 1), dtype=np.uint8); f[:42] = 200
    _write_demo(tmp_path / "fail.npz", np.stack([f]), [3], reached_exit=False)
    r = DemoRetriever(str(tmp_path))
    assert len(r) == 0                                   # the failed demo is excluded


def test_skip_noop_drops_idle_frames(tmp_path):
    f1 = np.zeros((84, 84, 1), dtype=np.uint8); f1[:42] = 200
    f2 = np.zeros((84, 84, 1), dtype=np.uint8); f2[:, :42] = 200
    _write_demo(tmp_path / "d.npz", np.stack([f1, f2]), [0, 5])  # frame1 = no-op
    r = DemoRetriever(str(tmp_path), skip_noop=True, noop_action=0)
    assert len(r) == 1 and r.actions[0] == 5             # only the active frame indexed


def test_empty_dir_is_safe(tmp_path):
    r = DemoRetriever(str(tmp_path))
    assert len(r) == 0
    assert r.retrieve(np.zeros((84, 84, 1), dtype=np.uint8), 0.9) == (None, 0.0)


def test_frame_encoder_trains_and_embeds(tmp_path):
    import numpy as np
    pytest.importorskip("torch")
    from rl.frame_encoder import train_frame_encoder, FrameEncoder, EMBED_DIM
    # Two distinct structured frames, flagged successful.
    f1 = np.zeros((84, 84, 1), dtype=np.uint8); f1[:42] = 200
    f2 = np.zeros((84, 84, 1), dtype=np.uint8); f2[:, :42] = 200
    obs = np.repeat(np.stack([f1, f2]), 8, axis=0)         # 16 frames
    _write_demo(tmp_path / "d.npz", obs, [0, 1] * 8)
    out = train_frame_encoder(str(tmp_path), out_path=str(tmp_path / "enc.pt"),
                              epochs=2, batch_size=8)
    enc = FrameEncoder(out)
    v = enc.embed(f1)
    assert v.shape == (EMBED_DIM,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-4)   # L2-normalised


def test_retriever_uses_encoder(tmp_path):
    import numpy as np
    pytest.importorskip("torch")
    from rl.frame_encoder import train_frame_encoder, FrameEncoder
    f1 = np.zeros((84, 84, 1), dtype=np.uint8); f1[:42] = 200
    f2 = np.zeros((84, 84, 1), dtype=np.uint8); f2[:, :42] = 200
    obs = np.repeat(np.stack([f1, f2]), 8, axis=0)
    _write_demo(tmp_path / "d.npz", obs, [3, 7] * 8)
    out = train_frame_encoder(str(tmp_path), out_path=str(tmp_path / "enc.pt"),
                              epochs=2, batch_size=8)
    r = DemoRetriever(str(tmp_path), encoder=FrameEncoder(out))
    assert r.descriptors.shape[1] == 64                    # learned-embedding width, not 144
    assert len(r) == 16
