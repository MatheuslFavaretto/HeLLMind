---
type: concept
id: concept_policy_entropy
created_step: 2500
tags:
  - RL
  - Entropia
  - Aprendizado por Reforço
  - Técnica de Treinamento
  - concept
---

# Policy Entropy

Policy entropy é uma métrica usada no domínio de aprendizado por reforço (RL) para medir a incerteza ou dispersão das probabilidades de ações escolhidas pelo agente. Em um ambiente como o Doom, onde o agente precisa tomar decisões complexas em cada etapa do jogo, a política de um agente pode se tornar menos entropica à medida que ele aprende e sua incerteza sobre as melhores ações diminui.

## No treino (Doom)

No treinamento de um agente jogando Doom, a policy entropy é uma métrica útil para avaliar o comportamento do agente. Ao início do treinamento, o agente pode escolher entre várias ações com probabilidades iguais, resultando em alta entropia. À medida que o agente aprende e sua política se torna mais especializada, as probabilidades de ações menos eficazes diminuem, reduzindo a entropia da política.

## Relacionados

- [[Concept - Reforço]]
- [[Concept - Aprendizado por Reforço]]
- [[Concept - Algoritmos de Aprendizado por Reforço]]
- [[Concept - Agente Inteligente]]
- [[Concept - Doom Ambiente]]
- [[Concept - Política de Agente]]
