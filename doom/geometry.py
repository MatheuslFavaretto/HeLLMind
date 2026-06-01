"""Lê a geometria do mapa do ViZDoom (segmentos de parede) para o minimapa real.

Com `set_sectors_info_enabled(True)`, cada estado traz `state.sectors`, e cada setor
tem `lines` (segmentos com x1,y1,x2,y2 e is_blocking). Juntando as linhas que bloqueiam
temos o contorno real do nível — é o que desenhamos como fundo do minimapa, com o
caminho do agente por cima (em vez de "quadrados soltos").
"""
from typing import List


def read_wall_segments(game, blocking_only: bool = True, max_segments: int = 5000) -> List[List[float]]:
    """Retorna [[x1,y1,x2,y2], ...] das paredes do mapa atual (mundo, mesmas unidades
    de POSITION_X/Y). Vazio se a info de setores não estiver disponível."""
    state = game.get_state()
    sectors = getattr(state, "sectors", None) if state is not None else None
    if not sectors:
        return []
    segs: List[List[float]] = []
    for sec in sectors:
        for ln in sec.lines:
            if blocking_only and not ln.is_blocking:
                continue
            segs.append([float(ln.x1), float(ln.y1), float(ln.x2), float(ln.y2)])
            if len(segs) >= max_segments:
                return segs
    return segs
