"""Loop de feedback (Passo 5): parse do frontmatter, defaults e ControlCallback."""
import os

from rl.control import (
    ControlCallback,
    _coerce,
    ensure_control_note,
    read_frontmatter,
)


def test_coerce_types():
    assert _coerce("true") is True
    assert _coerce("False") is False
    assert _coerce("5") == 5 and isinstance(_coerce("5"), int)
    assert _coerce("0.3") == 0.3 and isinstance(_coerce("0.3"), float)
    assert _coerce('"texto"') == "texto"


def test_ensure_is_idempotent(tmp_path):
    p = os.path.join(tmp_path, "00-index", "control.md")
    ensure_control_note(p, 0.15, 50000)
    # edita e garante que ensure NÃO sobrescreve o que o usuário mudou
    open(p, "a", encoding="utf-8").write("\nMARCA DO USUARIO\n")
    ensure_control_note(p, 0.99, 1)
    assert "MARCA DO USUARIO" in open(p, encoding="utf-8").read()


def test_read_frontmatter(tmp_path):
    p = os.path.join(tmp_path, "control.md")
    ensure_control_note(p, 0.2, 12345)
    fm = read_frontmatter(p)
    assert fm["stop_training"] is False
    assert fm["novelty_threshold"] == 0.2
    assert fm["write_every_steps"] == 12345


class _FakeDoc:
    novelty_threshold = 0.15
    write_every_steps = 50000


def _make_cb(path, doc=None):
    cb = ControlCallback(control_path=path, every_steps=1, doc_callback=doc)
    cb.num_timesteps = 10
    cb._next_check = 0
    return cb


def test_callback_stop(tmp_path):
    p = os.path.join(tmp_path, "control.md")
    ensure_control_note(p, 0.15, 50000)
    txt = open(p, encoding="utf-8").read().replace(
        "stop_training: false", "stop_training: true"
    )
    open(p, "w", encoding="utf-8").write(txt)
    assert _make_cb(p)._on_step() is False  # interrompe o treino


def test_callback_applies_live_changes(tmp_path):
    p = os.path.join(tmp_path, "control.md")
    ensure_control_note(p, 0.15, 50000)
    txt = open(p, encoding="utf-8").read()
    txt = txt.replace("novelty_threshold: 0.15", "novelty_threshold: 0.4")
    txt = txt.replace("write_every_steps: 50000", "write_every_steps: 8000")
    open(p, "w", encoding="utf-8").write(txt)

    doc = _FakeDoc()
    assert _make_cb(p, doc)._on_step() is True
    assert doc.novelty_threshold == 0.4
    assert doc.write_every_steps == 8000
