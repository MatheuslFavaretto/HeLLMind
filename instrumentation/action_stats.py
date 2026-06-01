"""Estatísticas sobre a distribuição de ações escolhidas pela política."""
from typing import Dict, List

import numpy as np


def action_distribution(counts: np.ndarray, button_names: List[str]) -> Dict[str, float]:
    """Fração de cada ação no total da janela."""
    total = counts.sum()
    if total == 0:
        return {name: 0.0 for name in button_names}
    return {button_names[i]: float(counts[i] / total) for i in range(len(button_names))}


def action_entropy(counts: np.ndarray) -> float:
    """Entropia (em nats) da distribuição empírica de ações.

    Alta = política explorando/variando; baixa = política colapsando numa ação.
    Útil para detectar quando o agente 'travou' num comportamento.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def max_entropy(n_actions: int) -> float:
    """Entropia máxima possível (distribuição uniforme), para normalizar."""
    return float(np.log(n_actions)) if n_actions > 1 else 0.0
