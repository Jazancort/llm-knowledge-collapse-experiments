> **⚠️ OBSOLETE** — These are the M2 initial results only (3 generations, r=16).
> The actual final results (rank ablation, 10 generations, multi-backbone) are in `PROJECT_STATUS.md`.

# Resultados Finais M2 — Síntese Experimental

**Data:** 2026-06-23

---

## Resultado Consolidado

### Retention Accuracy (G1 — Sintético, 3 seeds)

| Seed | Gen 0 | Gen 1 | Gen 2 | Gen 3 |
|---|---|---|---|---|
| 15 | 79/79 (100%) | 75/79 (94.9%) | 75/79 (94.9%) | 75/79 (94.9%) |
| 137 | 79/79 (100%) | 75/79 (94.9%) | 75/79 (94.9%) | 74/79 (93.7%) |
| 256 | 79/79 (100%) | 76/79 (96.2%) | 76/79 (96.2%) | 75/79 (94.9%) |

**Média Gen 3: 94.5% ± 0.7%**

### Transições G1 (Sintético) — Seed 15

| Gen | C→C | C→W | W→C | W→W |
|---|---|---|---|---|
| 1 | 75 | 4 | 4 | 117 |
| 2 | 79 | 0 | 0 | 121 |
| 3 | 79 | 0 | 0 | 121 |

**Congelamento total após Gen 1.**

### Transições G2 (Real — Shards Independentes)

| Gen | C→C | C→W | W→C | W→W |
|---|---|---|---|---|
| 1 | ? | 3 | 4 | ? |
| 2 | ? | 0 | 4 | ? |
| 3 | ? | 2 | 2 | ? |

**Fluxo contínuo — modelo continua oscilando na fronteira factual.**

### CKA-Factual (Layer 13) vs Gen 0

| Grupo | Gen 1 | Gen 2 | Gen 3 |
|---|---|---|---|
| G1 (Sintético) | 0.9836 | 0.9824 | 0.9826 |
| G2 (Real) | 0.9827 | ~0.98x | ~0.98x |

**CKA-Factual é idêntico entre G1 e G2 — mede adaptação, não recursão.**

### Adapter Health

| Métrica | Gen 1 | Gen 2 | Gen 3 |
|---|---|---|---|
| Effective Rank | 11.08 | 10.95 | 10.95 |

**Saudável. Sem rank collapse.**

---

## Descobertas Científicas

### 1. Knowledge Collapse NÃO ocorreu (regime QLoRA)

Sob PEFT com recursão de dados sintéticos, o modelo retém 94.5% ± 0.7% do conhecimento factual original ao longo de 3 gerações. Reproduzido em 3 seeds.

### 2. A diferença entre G1 e G2 é na DINÂMICA, não na geometria

- CKA é igual → ambos sofrem o mesmo "imposto de adaptação"
- Transições são radicalmente diferentes → G1 congela, G2 continua oscilando

### 3. Recursão sintética causa Ossificação Factual

O modelo treinado em seus próprios outputs para de aprender E de esquecer. Entra em loop de auto-confirmação. Zero transições factuais após Gen 1.

Dados reais mantêm plasticidade (transições C→W e W→C continuam ocorrendo).

### 4. CKA-Factual > CKA-Global (confirmado)

CKA-Factual detecta drift local que CKA-Global não vê. Mas NÃO distingue recursão de fine-tuning genérico no choque inicial.

---

## Status da Hipótese Original

| Hipótese | Status |
|---|---|
| H1: Instabilidade representacional precede degradação factual | **REFORMULADA** — CKA mede adaptação genérica, não detecta collapse |
| H2: ESI > baselines como preditor | **NÃO TESTÁVEL** — não houve collapse para predizer |
| H3: Stage B observável | **REJEITADA** — modelo ficou MENOS confiante, não mais |

### Nova Tese

**"Recursive Synthetic Fine-Tuning Causes Factual Ossification Under Low-Rank Adaptation"**

---

## Decisão: Próximo Passo

| Opção | Custo | O que ganha |
|---|---|---|
| M3-Sanity (Qwen 3B, Gen0→Gen1) | ~1h | Valida que o padrão não é artefato de escala |
| Início da redação | 0 GPU | Paper draft com dados atuais |

**Recomendação do GPT:** Fazer M3-Sanity primeiro (1h) para blindar contra "é só porque o modelo é pequeno". Depois iniciar redação.

**Recomendação do Gemini:** Fazer M3-Sanity e depois transitar para redação imediatamente.
