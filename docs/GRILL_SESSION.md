# Grill Session — Registro

**Data:** 2026-06-23  
**Participantes:** Julio Azancort + Kiro (co-design)  
**Duração:** 10 perguntas críticas  

---

## Resumo das perguntas e decisões

### P1: O que conta como "explicação"?

**Preocupação:** Attention ≠ explanation (Jain & Wallace, 2019).

**Decisão:** Attention rollout como principal + CKA como validação independente. Se ambos degradam antes da factualidade, não é artefato de uma técnica.

**Rejeitado:** Gradient × Input (custo 2x), SHAP (proibitivo), Logit Lens (fase 2).

---

### P2: Como separar collapse de catastrophic forgetting?

**Preocupação:** Fine-tuning iterativo com pouco dado pode causar forgetting simples, sem ser knowledge collapse.

**Decisão:** Três grupos — G1 (replacement), G2 (real-only), G3 (accumulation). Se G1 ≫ G2 em degradação, o efeito é collapse. Se G1 ≈ G2, é forgetting.

---

### P3: Protocolo de geração de dados sintéticos?

**Preocupação:** Memorização se mesmas perguntas são usadas para treino e avaliação.

**Decisão:** Protocolo (a) — perguntas fixas, respostas regeneradas. MAS com datasets separados: Training Seed ≠ Evaluation ≠ Probe. Elimina crítica de memorização.

---

### P4: Quantas amostras? Fine-tune sobre base ou sobre geração anterior?

**Preocupação:** 5k amostras é pouco para QLoRA com 4M parâmetros (overfit). Treinar sobre base ≠ processo recursivo.

**Decisão:** 
- Training Seed: 10-15k amostras
- Merge LoRA a cada geração, próxima geração parte do merged
- LR=1e-5, epochs=2, fixos (sem early stopping)
- Registrar train loss, gradient norm, parameter delta

---

### P5: Como medir factual accuracy de forma confiável?

**Preocupação:** Accuracy ruidosa invalida a detecção temporal.

**Decisão:** Exact match normalizado para factoid QA (NQ/TriviaQA/PopQA). TruthfulQA adiado. Auditoria humana (50/geração) + Cohen's Kappa.

**Rejeitado:** LLM-as-judge como principal, embedding similarity.

---

### P6: Como calcular CKA de forma comparável?

**Preocupação:** 32 camadas × pares de gerações = caos. Tokens como observações vs mean-pooled.

**Decisão:** 
- CKA linear (não RBF)
- Mean-pooled hidden states por amostra
- Blocos: early/middle/late
- Duas comparações: Gen 0 vs Gen t (posição) + Gen t vs Gen t+1 (velocidade)
- Resultado NÃO alega causalidade — alega precedência temporal + poder preditivo

---

### P7: Reproducibilidade — quantas seeds?

**Preocupação:** n=1 é evidência anedótica.

**Decisão:** 3 seeds mínimo (42, 137, 256). Report: curvas individuais + média + IC 95% bootstrap. Resultado principal é Lead Time (GC - GFW), não curvas visuais.

---

### P8: Qwen 1.5B é suficiente?

**Preocupação:** Modelo muito pequeno pode não exibir Stage B. Baseline factual baixa.

**Decisão:** Gemma 3 4B ou Qwen2.5-3B como principal. 1.5B apenas para validar pipeline (piloto). QLoRA 4-bit em 3-4B cabe na RTX 3070 com gradient accumulation.

---

### P9: Como medir fluência e confiança para demonstrar Stage B?

**Preocupação:** Se não demonstra fluência estável + accuracy caindo, não demonstra Stage B.

**Decisão:** 
- Fluência: external PPL (GPT-2 frozen) + D4 + syntactic completeness
- Confiança: avg token log-prob + predictive entropy
- H3 (Stage B) é EXPLORATÓRIA, não primária. Se não aparecer, artigo sobrevive.

---

### P10: Critérios de sucesso pré-definidos?

**Preocupação:** P-hacking narrativo.

**Decisão:** Critérios escritos antes de rodar:
- Lead Time ≥ 2: sucesso forte
- Lead Time ≈ 1: sucesso moderado
- Lead Time > 0 consistente: publicável
- Lead Time ≈ 0: hipótese rejeitada
- Lead Time < 0: hipótese falsificada

Threshold GFW definido na run piloto (M2) e congelado.

---

## Meta-observações da sessão

1. **O artigo sobrevive mesmo sem Stage B** — H1 é independente de H3.
2. **O resultado mais importante é o Lead Time, não as curvas** — número > visualização.
3. **A tabela comparativa ESI vs baselines é potencialmente a contribuição principal** — se ESI for melhor preditor, é a descoberta.
4. **Milestones protegem contra investimento desperdiçado** — M1A em 30min valida tudo antes de investir dias.
5. **O design é falsificável** — cenários de fracasso estão definidos e não há como "salvar" a hipótese com re-interpretação post-hoc.
