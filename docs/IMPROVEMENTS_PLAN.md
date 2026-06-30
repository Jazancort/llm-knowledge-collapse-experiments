# Melhorias Finais — Plano de Execução

## Prioridade 1 (impacto máximo)

### Item A: Figura conceitual do framework (pressure → threshold → regimes)
- **O que:** Diagrama visual mostrando rank/LR/exposure → pressure → threshold → 3 regimes
- **Onde:** Discussion §5.1, antes ou depois do texto sobre ETP
- **Como:** HTML → Playwright PNG (mesmo pipeline do Fig 7)
- **Impacto:** Transforma resultados dispersos em teoria visual

### Item B: Tabela "claim → evidence" (evidence matrix)
- **O que:** Tabela mapeando cada claim principal ao experimento que a suporta
- **Onde:** Final de §4 (antes de Discussion) ou início de §5
- **Impacto:** Reviewer vê rigor; cada claim tem respaldo explícito

### Item C: Explicitar que a contribuição é qualitativa (não quantitativa)
- **O que:** Frase repetida 2-3x: "The contribution is the existence of a sharp transition, not the numerical threshold itself"
- **Onde:** Abstract (já parcialmente), §5.1, §7 (Conclusion)
- **Impacto:** Mata metade das críticas sobre generalização

## Prioridade 2 (defesa forte)

### Item D: Tabela de regimes (resumo visual)
- **O que:** Tabela 3×4: Pressure / Retention / Distribution / Regime
- **Onde:** §4.4 (distributional signatures) ou §5.1
- **Impacto:** Leitor entende taxonomia em 20 segundos

### Item E: "Threats to validity" / parágrafo defensivo nas Limitations
- **O que:** Parágrafo explicando por que, apesar das limitações, o resultado é robusto
- **Onde:** Final de §6 (Limitations)
- **Impacto:** Muda o frame de "isso é fraco" para "isso é reproduzível"

### Item F: Tabela de normalizações tentadas e falhadas (resultado negativo)
- **O que:** Mini-tabela: normalization tried / result (all failed)
- **Onde:** §4.2 (cross-backbone) ou §6 (limitations)
- **Impacto:** Mostra rigor ("tentamos e não funcionou")

## Prioridade 3 (polimento)

### Item G: Engineering implications (lista condensada)
- **O que:** 4-5 bullet points práticos para engenheiros
- **Onde:** §5.4 (practical implications), como lista destacada
- **Impacto:** Conversa diretamente com EAAI

### Item H: Mostrar seed-independence visualmente
- **O que:** Mencionar mais explicitamente que 3 seeds → mesmo regime (já nos dados)
- **Onde:** §4.1, reforçar na prose (não precisa de figura nova)
- **Impacto:** Modesto (já está nos ranges)

## Experimento adicional (se houver tempo/GPU)

### Item I: Intervenção de exposição em Gemma 3
- **O que:** Repetir C5 (random 5% downsample) em Gemma 3 r=16 (boundary case)
- **Onde:** §4.5, adicionando um parágrafo
- **Impacto:** MUITO alto — prova que exposure axis é cross-backbone
- **Custo:** ~1-2h GPU (5 gens × 1 seed × Gemma 3)
- **Status:** Não obrigatório para submissão, mas seria o maior reforço possível

---

## Ordem de execução sugerida

1. Item C (frases qualitativas — 5 min, zero risco)
2. Item D (tabela de regimes — 10 min)
3. Item B (evidence matrix — 10 min)
4. Item E (threats to validity — 10 min)
5. Item F (normalizações falhadas — 5 min)
6. Item G (engineering implications — 5 min)
7. Item A (figura conceitual — 30 min, HTML+screenshot)
8. Item I (experimento Gemma 3 — decisão separada)
