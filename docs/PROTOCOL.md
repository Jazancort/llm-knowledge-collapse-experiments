# Protocolo Experimental

## Early Detection of Knowledge Collapse Through Explanation Stability Analysis in Recursive LLM Training

**Data de congelamento do design:** 2026-06-23  
**Pesquisador principal:** Julio Azancort  
**Hardware:** Ryzen 5600X, 64GB RAM, RTX 3070 8GB  

---

## 1. Pergunta de Pesquisa

> Existe informação nas representações internas e padrões de atenção de LLMs que permite prever degradação factual futura sob treinamento recursivo, com antecedência superior às métricas tradicionais (perplexidade, entropia, diversidade lexical)?

---

## 2. Hipóteses

### H1 — Primária

Instabilidade representacional (medida por CKA e attention rollout) precede degradação factual (medida por exact match accuracy) em modelos submetidos a treinamento recursivo com dados sintéticos.

**Critério de confirmação:** Lead Time > 0, consistente nas 3 seeds.

### H2 — Secundária

O Explanation Stability Index (ESI) tem poder preditivo superior a perplexity, entropy e TTR na previsão de Accuracy(t+1).

**Critério de confirmação:** Correlação ESI→Accuracy(t+1) > correlação de cada baseline individual.

### H3 — Exploratória

O fenômeno de Stage B (valley of dangerous competence) é observável: fluência e confiança se mantêm enquanto accuracy cai.

**Critério de confirmação:** Fluency Index estável por ≥1 geração após queda de accuracy de 10pp.

---

## 3. Critérios de Sucesso (pré-definidos, imutáveis)

| Lead Time médio | Interpretação |
|---|---|
| ≥ 2 gerações | Sucesso forte — artigo com contribuição clara |
| ≈ 1 geração | Sucesso moderado — publicável com framing adequado |
| > 0 consistente nas seeds | Publicável como resultado positivo |
| ≈ 0 | Hipótese principal rejeitada |
| < 0 | Hipótese falsificada |

---

## 4. Definições Operacionais

### GFW (Generation of First Warning)

Primeira geração em que ESI ultrapassa o threshold definido na run piloto.

**Método de definição do threshold:**
- Executar run piloto (M2: 3 gerações, 1 seed, G1)
- Calcular ESI nas gerações 0-2
- Threshold = mean(ESI_Gen0→Gen1) + 2σ
- Congelar. Nunca mais alterar.

### GC (Generation of Collapse)

Primeira geração em que factual accuracy cai **10 pontos percentuais** em relação à Gen 0.

Exemplo: se Gen 0 = 72%, GC ocorre quando accuracy ≤ 62%.

### Lead Time

```
Lead Time = GC - GFW
```

Este é o resultado principal do artigo.

---

## 5. Design Experimental

### 5.1 Modelo

| Papel | Modelo | Justificativa |
|---|---|---|
| Piloto (validação de pipeline) | Qwen2.5-1.5B-Instruct | Rápido, cabe com folga na GPU |
| Principal (experimento oficial) | Gemma 3 4B ou Qwen2.5-3B-Instruct | Baseline factual forte o suficiente para observar Stage B |

Decisão final do modelo principal será tomada após M1A, comparando baseline accuracy de ambos no Evaluation Set.

### 5.2 Fine-tuning

- **Método:** QLoRA 4-bit (NF4, double quantization)
- **LoRA rank:** 16
- **LoRA alpha:** 32
- **Target modules:** q_proj, k_proj, v_proj, o_proj
- **Learning rate:** 1e-5 (fixo em todas as gerações)
- **Epochs:** 2 (fixo em todas as gerações)
- **Batch size:** 2 (com gradient accumulation = 8, effective batch = 16)
- **Max sequence length:** 512
- **Warmup:** 5%
- **bf16:** sim

### 5.3 Evolução Recursiva

```
Gen 0: Modelo base (sem fine-tuning)
       ↓ avalia
       ↓ gera respostas sintéticas do Training Seed

Gen 1: Fine-tune (QLoRA) no modelo base com dados sintéticos de Gen 0
       ↓ merge LoRA → novo modelo completo
       ↓ avalia
       ↓ gera respostas sintéticas

Gen 2: Fine-tune no modelo de Gen 1 (merged) com dados sintéticos de Gen 1
       ↓ merge → novo modelo
       ↓ avalia
       ↓ gera
       ...

Gen N: Repetir até Stage C ou 10 gerações
```

**Decisão crítica:** merge do LoRA a cada geração. O modelo t+1 parte do modelo completo (merged) da geração t, não do base original. Isso garante que erros se acumulam nos pesos como a teoria prevê.

### 5.4 Grupos Experimentais

| Grupo | Dados na geração t | O que testa |
|---|---|---|
| **G1 — Replacement** | Apenas respostas sintéticas geradas pelo modelo da geração anterior | Recursive knowledge collapse (hipótese principal) |
| **G2 — Real-only** | Dados reais novos (subsample do Training Seed, mesmo tamanho) | Controle para catastrophic forgetting |
| **G3 — Accumulation** | Dados reais acumulados + sintéticos da geração anterior | Controle baseado na literatura (Gerstgrasser et al., 2024) |

**Hipótese nula a rejeitar:** H0: A degradação observada em G1 é explicada integralmente por catastrophic forgetting (G2 degrada igualmente).

### 5.5 Datasets

| Conjunto | Tamanho | Fontes | Uso |
|---|---|---|---|
| **Training Seed (A)** | 10.000-15.000 | Natural Questions + TriviaQA + PopQA | Gerar respostas sintéticas e treinar |
| **Evaluation Set (B)** | 1.000 | Mesmas fontes, disjunto de A | Medir accuracy, fluência, confiança |
| **Probe Set (C)** | 200 | Subconjunto fixo, nunca treina | Attention rollout, CKA, ESI |

**Regra fundamental:** B e C NUNCA entram no treinamento. Isso elimina a crítica de memorização.

### 5.6 Geração de Dados Sintéticos

- **Protocolo:** perguntas fixas do Training Seed, respostas regeneradas pelo modelo da geração atual
- **Prompt:** "Answer the following question in 5 words or less.\nQuestion: {q}\nAnswer:"
- **Temperature:** 0.7
- **Top-p:** 0.9
- **Max tokens:** 64

### 5.7 Seeds e Reproducibilidade

- **Mínimo:** 3 seeds completas (42, 137, 256)
- **Controlado:** data shuffling, LoRA initialization, generation sampling
- **Metadata salvo a cada geração:** seed, modelo, temperatura, LR, dataset hash, timestamp, gradient norm, parameter delta

---

## 6. Métricas

### 6.1 Factual Accuracy (Evaluation Set)

**NQ / TriviaQA / PopQA:**
- Exact match normalizado com aliases
- Normalização: lowercase, remoção de artigos (a/an/the), remoção de pontuação, trim
- Prompt força resposta curta (≤5 palavras)

**TruthfulQA (fase 2, adiado):**
- Ensemble de juízes congelados
- Concordância obrigatória entre juízes

**Auditoria humana:**
- 50 amostras aleatórias por geração
- Cohen's Kappa entre humano e sistema automático
- Meta: κ > 0.8

### 6.2 Confiança

- **Average token log-probability:** média de log p(token_i) sobre todos os tokens gerados
- **Predictive entropy:** H(p) = -Σ p·log(p) sobre o vocabulário, média por token

### 6.3 Fluência

- **External perplexity:** GPT-2 Medium (congelado) avaliando as respostas geradas
- **Distinct-4:** proporção de 4-grams únicos no corpus de respostas
- **Syntactic completeness:** taxa de respostas que terminam adequadamente

### 6.4 Explicabilidade (Probe Set)

**Attention Rollout:**
- Rollout multiplicativo com residual connection (0.5·attn + 0.5·identity)
- Calculado para todas as camadas
- Resultado: vetor (seq_len,) de atenção agregada ao input pelo último token

**CKA Linear:**
- Mean-pooled hidden states por amostra: (seq_len, hidden_dim) → (hidden_dim,)
- Resultado: (200, hidden_dim) por camada
- Blocos: early (1-N/3), middle (N/3-2N/3), late (2N/3-N)
- Comparação Gen 0 vs Gen t (distância acumulada) + Gen t vs Gen t+1 (velocidade)

**ESI (Explanation Stability Index):**
```
ESI_t = α · JS(rollout_t, rollout_{t+1}) + β · (1 - ρ(rollout_t, rollout_{t+1}))
```
- JS = Jensen-Shannon divergence ao quadrado
- ρ = Spearman rank correlation
- α = β = 0.5
- Calculado por amostra no Probe Set, depois agregado (média)

### 6.5 Baselines de Comparação

Para demonstrar que ESI é superior, comparar poder preditivo de:
- Perplexity (do próprio modelo)
- Predictive entropy
- Type-Token Ratio (TTR)
- Confidence (avg log-prob)

Método: correlação de cada métrica em t com Accuracy em t+1.

### 6.6 Dinâmica de Treinamento

Registrar a cada geração:
- Train loss (final)
- Validation loss
- Gradient norm (média do último epoch)
- Parameter delta norm: ||θ_t - θ_{t-1}||

---

## 7. Análise Estatística

### 7.1 Resultado Principal

Tabela de Lead Time por seed:

| Seed | GFW | GC | Lead Time |
|---|---|---|---|
| 42 | ? | ? | ? |
| 137 | ? | ? | ? |
| 256 | ? | ? | ? |

Reportar: média, IC 95% bootstrap.

### 7.2 Poder Preditivo

Regressão: predizer Accuracy(t+1) usando cada métrica em t.

| Métrica | Correlação com Accuracy(t+1) | p-value |
|---|---|---|
| Perplexity | ? | ? |
| Entropy | ? | ? |
| TTR | ? | ? |
| Confidence | ? | ? |
| Attention Stability | ? | ? |
| CKA (middle) | ? | ? |
| ESI | ? | ? |

### 7.3 Visualização

1. Mean ± 95% CI — Accuracy por geração (3 grupos)
2. Mean ± 95% CI — CKA por geração (3 grupos, blocos early/middle/late)
3. Mean ± 95% CI — ESI por geração (3 grupos)
4. Curvas individuais sobrepostas (transparência) para mostrar consistência qualitativa
5. Scatter: ESI(t) vs Accuracy(t+1)

---

## 8. Plano B (se H1 falhar)

Se Lead Time ≈ 0 em todas as seeds:

1. Analisar qual camada colapsa primeiro (CKA por camada individual)
2. Comparar dinâmicas G1 vs G2 vs G3 como resultado independente
3. Pivotar para: "quais métricas correlacionam melhor com degradação factual sob recursive training?"
4. Publicar resultado negativo se consistente (explicabilidade não fornece vantagem preditiva sobre métricas tradicionais)

---

## 9. Milestones de Implementação

| Milestone | Escopo | Pergunta | Tempo estimado |
|---|---|---|---|
| **M1A** | Carregar modelo, inferência, hidden states, attention, CKA, ESI, accuracy — SEM treinamento | Consigo medir tudo? | 30-60 min |
| **M1B** | 1 geração: QLoRA → merge → gerar → reavaliar | O pipeline recursivo funciona ponta a ponta? | 2-3h |
| **M2** | 3 gerações, 1 seed, G1 | Existe sinal? Definir threshold GFW. | 6-8h |
| **M3** | 10 gerações, 3 grupos, 3 seeds | Experimento completo. | 3-4 dias (lab) |

Regra: só avançar para M(n+1) se M(n) validar com sucesso.

---

## 10. Limitações Conhecidas

1. **Escala:** resultados em 3-4B não garantem generalização para modelos maiores. Argumento: se o efeito aparece em modelos menores (onde o espaço para Stage B é menor), é plausível que apareça em modelos maiores com Stage B mais extenso.

2. **Causalidade:** o experimento demonstra precedência temporal, não causalidade. O claim é "early warning signal", não "explicações causam o colapso".

3. **Attention ≠ explanation:** mitigado por usar CKA como validação independente. Se ambos degradam antes da factualidade, o efeito dificilmente é artefato de uma técnica específica.

4. **Modelo piloto vs principal:** resultados do piloto (1.5B) podem não se replicar no modelo principal (3-4B). É por isso que M1A/M1B usam o piloto e o experimento oficial usa o principal.

5. **Datasets em inglês:** não há benchmark brasileiro neste estudo. Será future work.

---

## 11. Conexão com o Survey

Este experimento valida empiricamente os conceitos centrais do artigo "Recursive Training Failures in Large Language Models: A Unified Taxonomy, Security Analysis, and Governance Framework":

- **Section 2.2 (Knowledge Collapse):** reprodução empírica da degradação three-stage
- **Section 4.1 (Data Accumulation vs Replacement):** G3 vs G1 testa diretamente o resultado de Gerstgrasser et al.
- **Section 8 (Structural Paradox):** se H1 é confirmada, demonstra que o paradoxo (capability = vulnerability) é observável ao nível representacional antes de se manifestar em outputs

O artigo experimental se posiciona como validação empírica e extensão do framework teórico, adicionando a contribuição original de detecção precoce via explicabilidade.
