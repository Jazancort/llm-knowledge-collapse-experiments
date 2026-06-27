> **⚠️ OBSOLETE** — This document describes metrics for the earlier ESI/Lead-Time hypothesis.
> The project pivoted to capacity-gated recursive QLoRA regime transitions.
> Active metrics are: K0 retention, effective rank, transition matrices.
> See `PROJECT_STATUS.md` for the current experimental narrative.

# Definições Formais das Métricas

---

## 1. Explanation Stability Index (ESI)

Métrica composta que quantifica a instabilidade dos padrões de atenção entre gerações consecutivas de um modelo recursivamente treinado.

### Definição

Para cada amostra i no Probe Set, dado o attention rollout da geração t (r_t^i) e da geração t+1 (r_{t+1}^i):

```
ESI_t^i = α · JS(r_t^i, r_{t+1}^i)² + β · (1 - ρ(r_t^i, r_{t+1}^i))
```

Onde:
- JS(p, q) = Jensen-Shannon divergence entre distribuições normalizadas
- ρ = Spearman rank correlation
- α = β = 0.5

### Agregação

```
ESI_t = (1/N) Σᵢ ESI_t^i
```

sobre as N=200 amostras do Probe Set.

### Interpretação

- ESI ≈ 0: explicações estáveis entre gerações (modelo estável)
- ESI ↑: explicações mudando (possível precursor de degradação)

### Threshold (GFW)

Definido na run piloto (M2):
```
threshold = mean(ESI_0→1, ESI_1→2) + 2σ
```

---

## 2. CKA Linear (Centered Kernel Alignment)

Mede similaridade representacional entre dois modelos dado o mesmo conjunto de inputs.

### Definição

Dados X ∈ ℝ^{n×p} (representações do modelo A) e Y ∈ ℝ^{n×q} (representações do modelo B), ambos centrados:

```
CKA(X, Y) = ||X^T Y||²_F / (||X^T X||_F · ||Y^T Y||_F)
```

### Preparação dos dados

1. Alimentar Probe Set (200 amostras) em ambos os modelos
2. Para cada amostra, extrair hidden state de cada camada: (seq_len, hidden_dim)
3. Mean-pool sobre seq_len: (hidden_dim,)
4. Stack: (200, hidden_dim) por camada

### Blocos de camadas

```
early:  camadas [0, N/3)
middle: camadas [N/3, 2N/3)
late:   camadas [2N/3, N)
```

CKA por bloco = média dos CKA das camadas individuais do bloco.

### Comparações

- **Distância acumulada:** CKA(Gen 0, Gen t) — quanto se afastou do ponto de partida
- **Velocidade de drift:** CKA(Gen t, Gen t+1) — quão rápido está mudando agora

### Interpretação

- CKA ≈ 1.0: representações muito similares
- CKA ↓: representações divergindo

---

## 3. Attention Rollout

Agrega informação de atenção de todas as camadas em um único vetor de relevância por token de entrada.

### Definição

Dado um modelo com L camadas, cada uma com attention matrix A_l ∈ ℝ^{seq×seq} (média sobre heads):

```
R_0 = I (identidade)
R_l = (0.5 · A_l + 0.5 · I) · R_{l-1}    para l = 1, ..., L
```

O rollout final é a última linha de R_L:
```
rollout = R_L[-1, :]
```

Isso representa quanto de atenção o último token (posição de geração) dedica a cada token de entrada.

### Uso no experimento

- Calcular rollout para cada amostra do Probe Set
- Comparar rollouts entre gerações usando JS divergence e rank correlation (→ ESI)

---

## 4. Factual Accuracy (Exact Match)

### Normalização

```python
def normalize(text):
    text = text.lower()
    text = remove_articles(text)      # "the", "a", "an"
    text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return text.strip()
```

### Match

```
correct = normalize(prediction) ∈ {normalize(gt) for gt in ground_truths}
```

Onde ground_truths é a lista de aliases aceitáveis (ex: ["Barack Obama", "Obama", "President Obama"]).

### Accuracy

```
Accuracy = (1/N) Σ correct_i
```

---

## 5. Confiança

### Average Token Log-Probability

```
Confidence = (1/T) Σ_{t=1}^{T} log p_θ(token_t | context_t)
```

Onde T é o número de tokens gerados.

**Interpretação:** valores mais altos (menos negativos) = modelo mais confiante.

### Predictive Entropy

```
H_t = -Σ_{v∈V} p_θ(v | context_t) · log p_θ(v | context_t)
```

Média sobre tokens gerados:
```
Entropy = (1/T) Σ_{t=1}^{T} H_t
```

**Interpretação:** menor entropia = distribuição mais concentrada = modelo mais decidido.

---

## 6. Fluência

### External Perplexity

Modelo externo congelado (GPT-2 Medium) avalia as respostas:

```
PPL_ext = exp(-(1/T) Σ log p_GPT2(token_t | context_t))
```

**Interpretação:** PPL baixa = texto fluente segundo modelo externo.

### Distinct-4

```
D4 = |{4-grams únicos}| / |{todos os 4-grams}|
```

Calculado sobre o corpus completo de respostas da geração.

**Interpretação:** D4 baixo = respostas repetitivas (sinal de model collapse).

---

## 7. Métricas de Dinâmica de Treinamento

### Parameter Delta Norm

```
Δθ_t = ||θ_t - θ_{t-1}||₂
```

Norma L2 da diferença entre parâmetros do modelo em gerações consecutivas.

### Gradient Norm

Norma média dos gradientes no último epoch de treinamento da geração t.

---

## 8. Resultado Principal: Lead Time

```
Lead Time = GC - GFW
```

Onde:
- GFW = primeira geração em que ESI > threshold
- GC = primeira geração em que Accuracy cai 10pp vs Gen 0

**Se Lead Time > 0:** ESI detectou instabilidade antes da degradação ser observável em accuracy.

---

## 9. Poder Preditivo Comparado

Para cada métrica M ∈ {Perplexity, Entropy, TTR, Confidence, Attention Stability, CKA, ESI}:

```
r_M = Pearson_correlation(M_t, Accuracy_{t+1})
```

**Nota:** correlação com Accuracy(t+1), não com Accuracy(t). Isso mede capacidade preditiva, não associação simultânea.

A tabela comparativa é potencialmente a contribuição principal do artigo:

| Métrica | r com Accuracy(t+1) | p-value |
|---|---|---|
| Perplexity | ? | ? |
| Entropy | ? | ? |
| TTR | ? | ? |
| Confidence | ? | ? |
| ESI | ? (espera-se o maior) | ? |
| CKA (middle) | ? | ? |
