# Registro de Decisões de Design

Cada decisão foi tomada durante a sessão de grill (2026-06-23) e está congelada para o experimento.

---

## D01: Modelo principal — Gemma 3 4B ou Qwen2.5-3B (não 1.5B)

**Contexto:** RTX 3070 8GB. Modelos maiores dão baseline factual mais forte, aumentando chance de observar Stage B.

**Decisão:** Usar 3-4B como modelo principal. 1.5B apenas para validar pipeline (piloto).

**Alternativa rejeitada:** Qwen2.5-1.5B como principal — risco alto de baseline factual muito baixa, Stage B invisível ou inexistente.

**Consequência:** Fine-tuning mais lento (~2x), mas tolerável. Offload para RAM (64GB) disponível se necessário.

---

## D02: Merge LoRA a cada geração (evolução recursiva real)

**Contexto:** Duas opções — (A) sempre treinar sobre o modelo base com novo adapter, ou (B) merge adapter → modelo resultante é o ponto de partida da próxima geração.

**Decisão:** Opção B — merge e continuar.

**Justificativa:** Opção A não estuda processo recursivo. Estuda "como datasets sintéticos diferentes afetam o mesmo modelo". Opção B garante que erros se acumulam nos pesos, que é o mecanismo central do knowledge collapse.

**Alternativa rejeitada:** Stacking de adapters (adapter sobre adapter) — cria artefato de adapter drift separado do fenômeno estudado.

---

## D03: Três grupos experimentais (Replacement, Real-only, Accumulation)

**Contexto:** Sem controle, não é possível separar knowledge collapse de catastrophic forgetting.

**Decisão:** 
- G1 (Replacement): apenas sintético — testa hipótese principal
- G2 (Real-only): apenas dados reais novos — controle para forgetting
- G3 (Accumulation): real + sintético acumulado — controle baseado na teoria

**Justificativa:** Se G1 ≫ G2 em degradação → efeito é recursive collapse, não forgetting. Se G1 ≈ G2 → hipótese comprometida.

---

## D04: Datasets separados (Training Seed ≠ Evaluation ≠ Probe)

**Contexto:** Se o mesmo conjunto é usado para treinar e avaliar, memorização confunde com collapse real.

**Decisão:**
- Training Seed (A): 10-15k — usado para gerar sintéticos e treinar
- Evaluation Set (B): 1k — medir accuracy, nunca treina
- Probe Set (C): 200 — extrair attention/CKA, nunca treina

**Defesa para reviewer:** "O conjunto de avaliação e o conjunto de sondagem explicativa nunca foram utilizados para treinamento, eliminando a possibilidade de que as tendências observadas sejam explicadas por simples memorização."

---

## D05: Attention rollout (principal) + CKA (validação)

**Contexto:** Attention ≠ explanation (Jain & Wallace, 2019). Reviewer pode atacar.

**Decisão:** Usar attention rollout como métrica principal de explicabilidade, com CKA como validação independente.

**Defesa:** "Se ambos (atenção e representações) degradam antes da factualidade, o efeito dificilmente é artefato de uma técnica específica de XAI."

**Alternativas rejeitadas:**
- Gradient × Input: dobra custo computacional, gera ruído alto token a token
- SHAP: computacionalmente proibitivo (10 gerações × 200 amostras × dezenas de tokens)
- Logit Lens: interessante mas deixado para fase 2

---

## D06: ESI como métrica única principal

**Contexto:** Com 3 hipóteses, 8 métricas e 20 gráficos, a narrativa ficaria difusa.

**Decisão:** ESI é A métrica principal. CKA e attention são componentes internos.

**Fórmula:**
```
ESI_t = 0.5 · JS(rollout_t, rollout_{t+1}) + 0.5 · (1 - ρ_t)
```

**Contribuição principal do artigo:** comparar ESI vs baselines na previsão de Accuracy(t+1).

---

## D07: Exact match normalizado (não LLM-as-judge) como métrica de accuracy

**Contexto:** Se a medição de accuracy é ruidosa, a detecção temporal (Lead Time) fica impossível.

**Decisão:** Exact match com aliases normalizados para NQ/TriviaQA/PopQA. TruthfulQA adiado para fase 2.

**Justificativa:** 
- Reproduzível sem API externa
- Sem viés de juiz
- Baixo ruído
- Auditoria humana (50/geração) valida concordância

**Alternativas rejeitadas:**
- LLM-as-judge como principal: introduz outra IA dentro do experimento, cria dependência
- Embedding similarity: não distingue "Paris" de "Lyon" (embeddings próximos, respostas opostas)

---

## D08: Hiperparâmetros fixos (LR=1e-5, epochs=2)

**Contexto:** Se cada geração para em pontos diferentes (via early stopping), o efeito observado pode vir do número de updates, não da qualidade dos dados.

**Decisão:** LR e epochs fixos para todas as gerações e grupos. A única variável que muda é a qualidade dos dados.

**Consequência:** Pode haver gerações com underfitting ou overfitting relativo. Aceitável — o objetivo é isolar o efeito dos dados, não otimizar performance por geração.

---

## D09: Threshold GFW definido na run piloto e congelado

**Contexto:** Se o threshold é definido post-hoc, existe risco de p-hacking.

**Decisão:** 
1. Rodar M2 (3 gerações, 1 seed, G1)
2. Calcular ESI nas gerações 0→1 e 1→2
3. Threshold = mean + 2σ
4. Congelar. Nunca alterar.
5. Aplicar no experimento completo (M3)

---

## D10: GC = queda de 10 pontos percentuais (não relativa)

**Contexto:** 10% relativo vs 10pp absoluto são coisas diferentes.

**Decisão:** 10 pontos percentuais absolutos em relação à Gen 0.

**Exemplo:** Gen 0 = 72% → GC quando accuracy ≤ 62%.

**Justificativa:** Mais interpretável, independente do baseline.

---

## D11: 3 seeds mínimo

**Contexto:** Uma única seed pode produzir resultado anedótico (n=1).

**Decisão:** 3 runs completas com seeds 42, 137, 256. Escalável no lab se necessário.

**Reporting:** Curvas individuais + média + IC 95% bootstrap. Tabela de Lead Time por seed.

---

## D12: H1 como primária, Stage B como exploratória

**Contexto:** Stage B pode não aparecer em modelos de 3-4B (modelos menores podem não ter capacidade suficiente para manter fluência enquanto perdem factualidade).

**Decisão:** H1 (instabilidade precede degradação) é a hipótese primária. H2 (ESI > baselines) é secundária. H3 (Stage B) é exploratória.

**Consequência:** Se Stage B não aparecer, o artigo sobrevive. Se H1 falhar, o artigo precisa de pivot.

---

## D13: TruthfulQA adiado para fase 2

**Contexto:** TruthfulQA requer LLM-as-judge (respostas abertas). O ensemble (Qwen-7B + Llama-8B) é pesado na 3070.

**Decisão:** Não incluir TruthfulQA no M1-M3. Artigo se sustenta com NQ + TriviaQA + PopQA.

**Justificativa:** Factoid QA com exact match é mais limpo, mais rápido, mais defensável.

---

## D14: Implementação em milestones progressivos

**Contexto:** Risco de investir dias em pipeline que falha em algum ponto.

**Decisão:**
- M1A: validação de infraestrutura (sem treinamento) — 30-60 min
- M1B: ciclo mínimo (1 geração) — 2-3h
- M2: 3 gerações para calibrar threshold — 6-8h
- M3: experimento completo — 3-4 dias

**Regra:** só avança se o milestone anterior passar.

---

## D15: Temperature 0.7 para geração sintética

**Contexto:** Temperature muito baixa (0) acelera colapso artificialmente (distribução degenerada). Temperature muito alta (1.0+) pode gerar respostas ruins desde Gen 0.

**Decisão:** 0.7 como default. Sensibilidade a T=1.0 como possível experimento complementar (não obrigatório).
