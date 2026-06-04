"""Tests for rl.bc — behavioral cloning pure pieces (button mapping, demo IO, BC loss)."""
import numpy as np

from rl.bc import bc_cross_entropy, load_demos, nearest_action, save_demo


# --------------------------- nearest_action ---------------------------
# Button order (CAMPAIGN_BUTTONS): FWD, BACK, TL, TR, ATK, USE, SPEED, NEXTW, ML, MR

def test_nearest_action_exact_match():
    actions = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # FWD
               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]   # FWD+ATK
    pressed = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]     # human held forward + attack
    assert nearest_action(pressed, actions) == 1


def test_nearest_action_prefers_more_overlap():
    actions = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # ATK only
               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]   # FWD+ATK
    pressed = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]     # fwd+atk -> the combo, not bare ATK
    assert nearest_action(pressed, actions) == 1


def test_nearest_action_punishes_extra_buttons():
    actions = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # FWD
               [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]]   # FWD+TL+ATK
    pressed = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # just forward -> plain FWD
    assert nearest_action(pressed, actions) == 0


def test_nearest_action_no_press_picks_minimal():
    actions = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    # nothing pressed -> whichever adds least; both add 1 extra, first wins by stability
    assert nearest_action([0] * 10, actions) in (0, 1)


# --------------------------- demo IO ---------------------------

def test_save_then_load_round_trip(tmp_path):
    obs = np.zeros((5, 84, 84, 1), dtype=np.uint8)
    acts = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    save_demo(str(tmp_path / "demo_000.npz"), obs, acts)
    lo, la = load_demos(str(tmp_path))
    assert lo.shape == (5, 84, 84, 1)
    assert list(la) == [0, 1, 2, 1, 0]


def test_load_demos_concatenates(tmp_path):
    save_demo(str(tmp_path / "a.npz"), np.zeros((3, 4, 4, 1), np.uint8), np.array([0, 1, 0]))
    save_demo(str(tmp_path / "b.npz"), np.zeros((2, 4, 4, 1), np.uint8), np.array([2, 2]))
    _, la = load_demos(str(tmp_path))
    assert len(la) == 5


def test_load_demos_empty_dir(tmp_path):
    obs, acts = load_demos(str(tmp_path))
    assert len(obs) == 0 and len(acts) == 0


# --------------------------- BC loss ---------------------------

def test_bc_cross_entropy_decreases_with_correct_logits():
    import torch
    targets = torch.tensor([0, 1, 2])
    # confident-correct logits -> low loss; uniform -> higher loss
    good = torch.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
    flat = torch.zeros((3, 3))
    assert float(bc_cross_entropy(good, targets)) < float(bc_cross_entropy(flat, targets))


def test_bc_cross_entropy_one_grad_step_reduces_loss():
    import torch
    torch.manual_seed(0)
    net = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(net.parameters(), lr=0.1)
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    with torch.no_grad():
        before = float(bc_cross_entropy(net(x), y))
    for _ in range(5):
        opt.zero_grad()
        bc_cross_entropy(net(x), y).backward()
        opt.step()
    with torch.no_grad():
        after = float(bc_cross_entropy(net(x), y))
    assert after < before
