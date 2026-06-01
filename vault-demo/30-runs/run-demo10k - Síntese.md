---
type: synthesis
run: run-demo10k
created: 2026-06-01T08:21:13+00:00
checkpoints: 4
tags:
  - synthesis
  - doom-rl
---

# Síntese da Run PPO no Doom - run-demo10k

**Run:** [[run-demo10k]]

Ao longo dos primeiros 2.500 passos, o agente começa com um desempenho baixo, obtendo uma recompensa de apenas 106.3 e não conseguindo nenhuma mira ou eliminação. No entanto, a política inicial parece bem exploratória, com alta entropia indicando uma variedade de ações tomadas. A partir dos 2.500 passos até os 5.000 passos, o agente começa a mostrar sinais de aprendizado, com a precisão saltando para 1% e a recompensa caindo para 79.8, indicando que ele está começando a se mover e a interagir com o ambiente. No entorno dos 7.500 passos, a política se torna um pouco menos exploratória, com a entropia diminuindo para 1%, mas o agente está agora conseguindo mais eliminações (0.8 kills/ep) e uma recompensa de 50.8. A partir dos 7.500 passos até os 10.000 passos, há um pequeno aumento na recompensa para 104.4, mas a precisão permanece estável em 1%, sugerindo que o agente ainda não dominou a mira precisa. A entropia diminuiu ligeiramente para 0.99, indicando uma política mais especializada, e a distância percorrida por episódio caiu significativamente, sugerindo um melhor controle do movimento.

## Marcos do treino

- ~2,500 steps: Alta entropia, baixa precisão e recompensa inicial
- ~5,000 steps: Primeiros sinais de aprendizado com aumento na precisão e eliminações
- ~7,500 steps: Entropia diminui, melhor controle do movimento e mais eliminações por episódio
- ~10,000 steps: Pequeno aumento na recompensa sem mudança significativa na precisão

## Conceitos centrais

_(nenhum)_
