> **⚠️ OBSOLETE** — This document describes the earlier ESI/Lead-Time hypothesis.
> The project pivoted to capacity-gated recursive QLoRA regime transitions.
> See `PROJECT_STATUS.md` for the current experimental narrative.

# Fundamento Teórico e Conexão com o Survey

---

## Artigo de origem

**"Recursive Training Failures in Large Language Models: A Unified Taxonomy, Security Analysis, and Governance Framework"**  
Julio Leite Azancort Neto, André Carlos Ponce de Leon Ferreira de Carvalho, Carlos Renato Lisboa Francês  
Submetido para Springer (2026)

---

## Gap que este experimento preenche

O survey identifica explicitamente como future work (Section 9):

> "First, the empirical validation of collapse detection metrics in transformer-scale models: the theoretical results [...] require translation into practical benchmarks that can detect Stage B knowledge collapse in production systems without relying on the fluency and coherence metrics that the present analysis demonstrates to be inadequate."

Este experimento é exatamente essa validação empírica.

---

## Resultados teóricos que sustentam o experimento

### 1. Strong Model Collapse (Dohmatob et al., 2025)

```
E[L_T] ≥ E[L_0] + α·k·T
```

Implicação: qualquer fração k>0 de dados sintéticos causa crescimento linear do erro. Nosso G1 (replacement) deve exibir este comportamento.

### 2. Dirac Mass Convergence (Seddik et al., 2024)

```
lim_{T→∞} p_T(·) = δ_{τ*}
```

Implicação: sem dados reais, o modelo eventualmente colapsa para output degenerado. Nosso G1 deve convergir para respostas homogêneas (D4 → 0).

### 3. Accumulation Bound (Gerstgrasser et al., 2024)

```
MSE_T ≤ (σ²d)/(T-d-1) · π²/6
```

Implicação: acumulação mantém erro bounded. Nosso G3 deve exibir estabilidade significativamente maior que G1.

### 4. Random Walk in Parameter Space (Xu et al., 2025)

```
P(T) < 1/2
```

Implicação: probabilidade de melhoria via training recursivo é sempre < 50%. O drift representacional que medimos com CKA é a manifestação observável deste random walk.

### 5. Three-Stage Knowledge Collapse (Keisha et al., 2025)

- Stage A: knowledge preservation
- Stage B: valley of dangerous competence (accuracy ↓, fluency ≈)
- Stage C: instruction collapse

Nossa H3 testa se Stage B é observável em escala reduzida.

### 6. H-Neurons (Gao et al., 2025)

Neurônios esparsos (<0.1% dos parâmetros) predizem alucinação com alta accuracy. Codificam "over-compliance" — disposição a satisfazer requests mesmo quando factualmente incorreto.

Conexão: se CKA nas camadas intermediárias (onde H-neurons residem) diverge antes da accuracy cair, estamos potencialmente observando a ativação progressiva desse circuito.

---

## Contribuição original deste experimento

O survey é teórico. Este experimento adiciona:

1. **Reprodução empírica** de knowledge collapse em escala reduzida (nenhum paper existente demonstra os 3 estágios com métricas granulares por geração)

2. **Explicabilidade como early warning** — pergunta nunca feita na literatura:
   > "A instabilidade das explicações precede a degradação factual?"

3. **ESI como métrica operacional** — proposta de índice quantificável que pode ser monitorado em produção

4. **Comparação G1/G2/G3** — separação experimental limpa entre collapse e forgetting (gap na literatura)

5. **Quantificação do Lead Time** — resultado numérico sobre antecedência de detecção

---

## Posicionamento na literatura

| Trabalho | O que demonstra | O que NÃO faz |
|---|---|---|
| Shumailov et al., 2024 | Model collapse existe (Nature) | Não mede explicabilidade, não detecta Stage B |
| Dohmatob et al., 2025 | Collapse é linear com k (ICLR) | Teórico, sem experimento de detecção |
| Keisha et al., 2025 | 3 estágios de knowledge collapse | Não propõe mecanismo de detecção precoce |
| Gerstgrasser et al., 2024 | Accumulation previne collapse | Não compara com métricas de explicabilidade |
| Gao et al., 2025 | H-Neurons predizem alucinação | Não conecta com recursive training |
| **Este trabalho** | **ESI detecta collapse antes de accuracy** | Não demonstra causalidade, escala limitada |

---

## Venues alvo (por cenário de resultado)

**Resultado forte (Lead Time ≥ 2):**
- AAAI (AI Safety track)
- AISTATS
- NeurIPS (workshops: Trustworthy ML, Foundation Models)
- Expert Systems with Applications
- Information Sciences

**Resultado moderado (Lead Time ≈ 1):**
- Engineering Applications of AI
- Applied Intelligence
- Neurocomputing
- Conference workshops (ICML, NeurIPS)

**Resultado negativo (publicável como negativo):**
- Workshops de ML Reliability
- Journal of Machine Learning Research (short papers / negative results)
