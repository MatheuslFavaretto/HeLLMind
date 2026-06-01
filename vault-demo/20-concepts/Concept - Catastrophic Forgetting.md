---
type: concept
id: concept_catastrophic_forgetting
created_step: 7500
tags:
  - concept
---

# Catastrophic Forgetting

Catastrophic Forgetting é um fenômeno onde um agente de aprendizado por reforço (RL) perde habilidades ou conhecimentos adquiridos anteriormente ao se focar na aquisição de novas competências.

## No treino (Doom)

No contexto do jogo Doom, considerando o treinamento de um agente, a catastrophic forgetting pode ocorrer quando o agente é exposto a uma nova fase ou nível com requisitos diferentes. Por exemplo, se inicialmente o agente foi treinado para navegar eficientemente em um ambiente com poucos inimigos, e então é transferido para um cenário mais desafiador com muitos inimigos, ele pode começar a esquecer as habilidades básicas de navegação adquiridas no nível anterior. Isso resulta na regressão do agente em tarefas que ele já dominava, mesmo enquanto aprende novas estratégias para lidar com o cenário mais complexo.

## Relacionados

- [[Concept - Aprendizado por Reforço]]
- [[Concept - Transferência de Aprendizagem]]
- [[Concept - Memória Sequencial]]
