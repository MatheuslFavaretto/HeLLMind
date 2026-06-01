"""Loop de feedback: o Obsidian deixa de ser só receptor e passa a CONTROLAR.

Cria uma nota de controle em `00-index/control.md` com um frontmatter YAML simples.
A cada N steps o treino relê esse arquivo e se adapta SEM reiniciar:
- `stop_training: true`  -> encerra o treino de forma limpa.
- `novelty_threshold`    -> mais/menos sensível para escrever notas (ao vivo).
- `write_every_steps`    -> cadência de coleta de snapshots (ao vivo).

Mantemos um parser minúsculo (sem dependência de YAML): o frontmatter é só
`chave: valor` entre `---`, o que basta para um painel de controle.
"""
import os
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback


def _coerce(v: str) -> Any:
    s = v.strip().strip('"').strip("'")
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def read_frontmatter(path: str) -> Dict[str, Any]:
    """Lê o frontmatter YAML simples de uma nota. Tolerante a erros (retorna {})."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    out: Dict[str, Any] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = _coerce(val)
    return out


def ensure_control_note(
    path: str, novelty_threshold: float, write_every_steps: int
) -> None:
    """Cria a nota de controle com defaults, se ela ainda não existir."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = (
        "---\n"
        "type: control\n"
        "stop_training: false\n"
        f"novelty_threshold: {novelty_threshold}\n"
        f"write_every_steps: {write_every_steps}\n"
        "---\n\n"
        "# Painel de controle do treino\n\n"
        "Edite os valores do **frontmatter** acima enquanto o treino roda — ele relê\n"
        "este arquivo a cada poucos milhares de steps e se adapta sem reiniciar.\n\n"
        "- `stop_training: true` encerra o treino de forma limpa (salva o modelo).\n"
        "- `novelty_threshold` controla o quão diferente algo precisa ser p/ virar nota.\n"
        "- `write_every_steps` controla a cadência de coleta.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


class ControlCallback(BaseCallback):
    """Relê a nota de controle periodicamente e aplica as mudanças no treino."""

    def __init__(
        self,
        control_path: str,
        every_steps: int,
        doc_callback: Optional[BaseCallback] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.control_path = control_path
        self.every_steps = max(1, every_steps)
        self.doc_callback = doc_callback
        self._next_check = 0

    def _on_training_start(self) -> None:
        nt = getattr(self.doc_callback, "novelty_threshold", 0.15)
        we = getattr(self.doc_callback, "write_every_steps", 50000)
        ensure_control_note(self.control_path, nt, we)
        if self.verbose:
            print(f"[control] painel em {self.control_path} (edite p/ controlar o treino)")

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_check:
            return True
        self._next_check = self.num_timesteps + self.every_steps

        ctrl = read_frontmatter(self.control_path)
        if not ctrl:
            return True

        if ctrl.get("stop_training") is True:
            print(f"[control] stop_training=true — encerrando em {self.num_timesteps} steps.")
            return False  # SB3 interrompe o learn() de forma limpa

        if self.doc_callback is not None:
            nt = ctrl.get("novelty_threshold")
            if isinstance(nt, (int, float)) and nt != self.doc_callback.novelty_threshold:
                if self.verbose:
                    print(f"[control] novelty_threshold -> {nt}")
                self.doc_callback.novelty_threshold = float(nt)
            we = ctrl.get("write_every_steps")
            if isinstance(we, int) and we > 0 and we != self.doc_callback.write_every_steps:
                if self.verbose:
                    print(f"[control] write_every_steps -> {we}")
                self.doc_callback.write_every_steps = int(we)
        return True
