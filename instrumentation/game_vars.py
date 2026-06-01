"""Definição das GameVariables do ViZDoom que vamos extrair a cada passo.

A ideia central: capturar MUITO sinal além do reward. Separamos as variáveis em:
- MONOTONIC: contadores que só sobem dentro de um episódio (kills, dano, etc).
  Para essas, reportamos o DELTA por passo — assim podemos somar ao longo de uma
  janela mesmo com episódios resetando no meio.
- LEVELS: valores instantâneos (vida, munição). Reportamos o valor atual e
  amostramos para tirar média/mínimo na janela.
"""
import vizdoom as vzd

# A ORDEM aqui define a ordem de `state.game_variables`. Não reordene sem ajustar.
TRACKED_VARS = [
    vzd.GameVariable.KILLCOUNT,       # inimigos mortos
    vzd.GameVariable.HITCOUNT,        # tiros que acertaram
    vzd.GameVariable.HITS_TAKEN,      # vezes que o agente foi atingido
    vzd.GameVariable.DAMAGECOUNT,     # dano total causado
    vzd.GameVariable.DAMAGE_TAKEN,    # dano total tomado
    vzd.GameVariable.DEATHCOUNT,      # mortes
    vzd.GameVariable.ITEMCOUNT,       # itens pegos
    vzd.GameVariable.HEALTH,          # vida atual (nível)
    vzd.GameVariable.AMMO2,           # munição da arma inicial (nível)
    vzd.GameVariable.POSITION_X,      # posição no mapa (p/ caminho/cobertura)
    vzd.GameVariable.POSITION_Y,      # posição no mapa (p/ caminho/cobertura)
    vzd.GameVariable.SELECTED_WEAPON, # arma selecionada (slot)
]

VAR_NAMES = [
    "killcount", "hitcount", "hits_taken", "damagecount", "damage_taken",
    "deathcount", "itemcount", "health", "ammo2",
    "position_x", "position_y", "selected_weapon",
]

# Contadores cumulativos -> reportamos delta por passo
MONOTONIC = [
    "killcount", "hitcount", "hits_taken", "damagecount", "damage_taken",
    "deathcount", "itemcount",
]
# Valores instantâneos -> reportamos nível atual (média/mín na janela)
LEVELS = ["health", "ammo2", "position_x", "position_y", "selected_weapon"]

assert len(TRACKED_VARS) == len(VAR_NAMES)
