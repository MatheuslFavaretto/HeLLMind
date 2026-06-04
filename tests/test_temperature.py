"""Tests for rl.eval._tempered_actions — tempered sampling (rescues the argmax-collapse)."""
import torch

from rl.eval import _tempered_actions


class _FakeDist:
    def __init__(self, logits):
        self.distribution = type("D", (), {"logits": logits})()


class _FakePolicy:
    """Mimics just the two methods _tempered_actions touches."""
    def __init__(self, logits):
        self._logits = logits

    def obs_to_tensor(self, obs):
        return obs, None

    def get_distribution(self, obs_t):
        return _FakeDist(self._logits)


class _FakeModel:
    def __init__(self, logits):
        self.policy = _FakePolicy(logits)


def test_returns_valid_action_indices():
    # 3 actions, batch of 1
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    acts = _tempered_actions(_FakeModel(logits), obs=None, temperature=0.5)
    assert acts.shape == (1,)
    assert 0 <= int(acts[0]) <= 2


def test_low_temperature_concentrates_on_argmax():
    # action 1 has the highest logit; at T→0 sampling should pick it ~always.
    logits = torch.tensor([[0.0, 5.0, 0.0]])
    model = _FakeModel(logits)
    picks = [int(_tempered_actions(model, None, temperature=0.05)[0]) for _ in range(50)]
    assert picks.count(1) >= 48   # nearly always the argmax action


def test_high_temperature_spreads_actions():
    # Equal-ish logits at T=1 should NOT collapse to one action over many samples.
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 1.1, 0.9]])
    model = _FakeModel(logits)
    picks = {int(_tempered_actions(model, None, temperature=1.0)[0]) for _ in range(50)}
    assert len(picks) >= 2        # explores more than one action


def test_temperature_zero_is_safe():
    # max(T, 1e-6) guard means T=0 doesn't divide-by-zero; it behaves like argmax.
    logits = torch.tensor([[0.0, 9.0, 0.0]])
    acts = _tempered_actions(_FakeModel(logits), None, temperature=0.0)
    assert int(acts[0]) == 1
