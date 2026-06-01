"""Análise de regressão entre checkpoints (feature D).

Faz a documentação INTERPRETAR, não só descrever: quando uma métrica-chave despenca
de um checkpoint para o outro, isso costuma indicar que o agente "esqueceu" algo que
já sabia (Catastrophic Forgetting). Esta função pura detecta essas quedas para o
NoteWriter destacar na nota e linkar o conceito automaticamente.
"""
from typing import Dict, List, Optional

# Conceito que será criado/linkado quando houver regressão.
FORGETTING_CONCEPT = "Catastrophic Forgetting"
FORGETTING_DESCRIPTION = (
    "Quando uma rede perde habilidades já aprendidas ao otimizar para algo novo — "
    "no RL, o agente regride em métricas que antes dominava."
)

# (chave no snapshot, rótulo legível, é porcentagem?)
_WATCH = [
    ("shooting_accuracy", "precisão de tiro", True),
    ("mean_reward", "recompensa média", False),
    ("kills_per_episode", "kills/episódio", False),
    ("success_rate", "taxa de sucesso", True),
]

# Queda relativa (vs. o checkpoint anterior) a partir da qual marcamos regressão.
REGRESSION_DROP = 0.30


def detect_regressions(
    current: Dict, previous: Optional[Dict], threshold: float = REGRESSION_DROP
) -> List[str]:
    """Retorna descrições das métricas que caíram >= `threshold` (vazio se nada)."""
    if not previous:
        return []
    out: List[str] = []
    for key, label, pct in _WATCH:
        cur = float(current.get(key, 0.0))
        prev = float(previous.get(key, 0.0))
        if prev <= 1e-6:  # sem base positiva para falar em "queda"
            continue
        drop = (prev - cur) / prev
        if drop >= threshold:
            cf = f"{cur:.0%}" if pct else f"{cur:,.2f}"
            pf = f"{prev:.0%}" if pct else f"{prev:,.2f}"
            out.append(f"{label} caiu de {pf} para {cf} (−{drop:.0%})")
    return out
