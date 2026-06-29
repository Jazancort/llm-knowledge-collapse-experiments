# Future Work — Próximo Artigo

Ideias, experimentos e extensões que surgiram durante o projeto atual mas ficaram fora do escopo. Organizadas por potencial de impacto e viabilidade.

---

## Tier 1 — Alto impacto, extensões diretas deste trabalho

### 1.1 Step function vs gradual: mapeamento fino da critical boundary

**Pergunta:** A transição de ~5% menos dados que desloca o regime é um step function ou gradual?

**Experimento:** Variar a fração de dados removidos de 0% a 20% em incrementos de 2%. Plotar retention vs fração removida. Verificar se existe um threshold abrupto.

**Impacto:** Se for step function, é um fenômeno de phase transition — muito mais interessante teoricamente.

### 1.2 TCE Loss + QLoRA combinado

**Pergunta:** Truncated Cross-Entropy (Zibakhsh 2024) opera no eixo de loss-signal; QLoRA opera no eixo de update-subspace. Combinados, quanto mais robusto fica?

**Experimento:** Rodar r=256 com TCE loss. Se TCE sozinho previne degradação em r=256, mostra que os dois mecanismos são ortogonais e cumulativos.

**Impacto:** Paper de mitigação prática. Muito publicável.

### 1.3 Subspace overlap analysis (CKA/principal angles entre deltas)

**Pergunta:** FFT e QLoRA perdem fatos diferentes (Jaccard=0.0). Isso reflete subespaços de atualização ortogonais?

**Experimento:** Computar CKA entre ΔW_FFT e B@A do QLoRA. Analisar principal angles. Verificar se fatos perdidos correlacionam com camadas/módulos mais alterados.

**Impacto:** Transformaria o achado item-level (output measure) em mecanismo weight-level. Paper de interpretabilidade.

### 1.4 Adaptive rank controller (governança automática)

**Pergunta:** Pode-se monitorar SDI/content-efficiency em tempo real e reduzir rank quando drift aparece?

**Experimento:** Controller que começa com r=256 e reduz para r=64 quando SDI > threshold. Comparar com rank fixo.

**Impacto:** Paper de governança/safety aplicada. Altamente relevante para produção.

### 1.5 Escala: 7B, 13B, 70B

**Pergunta:** O regime transition existe em modelos maiores? O threshold muda como?

**Experimento:** Repetir dose-response (r=4/16/64/256) em Qwen-7B ou Llama-3-8B. Pelo menos Gen5.

**Impacto:** Generalização é a maior limitação do paper atual. Se confirmar em 7B+, vira paper de alto impacto.

---

## Tier 2 — Médio impacto, mecanístico

### 2.1 Dataset entropy como preditor do threshold

**Pergunta:** O threshold depende da complexidade intrínseca dos dados de treino?

**Experimento:** Repetir o protocolo com datasets de complexidades diferentes (TriviaQA vs WikiText vs código vs math). Verificar se threshold varia com a complexidade.

**Impacto:** Se threshold = f(dataset_complexity), unifica a explicação de por que Gemma e Qwen diferem.

### 2.2 Logit divergence como early warning

**Pergunta:** A distribuição de logits diverge antes da retention cair?

**Experimento:** Salvar top-k logits por item por geração. Computar KL/JS divergence entre Gen0 e GenT. Verificar se precede queda de retention.

**Impacto:** Revive parcialmente a ideia de early warning que o ESI tentou resolver.

### 2.3 H-Neurons e recursive training

**Pergunta:** Os H-Neurons (Gao 2025) se ativam mais no regime degradativo?

**Experimento:** Extrair ativações de FFN layers, treinar classificador de hallucination-neurons, verificar se activation patterns mudam entre homeostático e degradativo.

**Impacto:** Conecta collapse com hallucination — bridge entre duas linhas de pesquisa.

### 2.4 Causalidade via intervention em exemplos ESPECÍFICOS

**Pergunta:** Se identificamos os ~121 exemplos longos que C3 remove, e removemos APENAS esses e verificamos que não importa quais se remove, isso confirma boundary-sensitivity vs quality.

**Experimento:** 
- C6: remove os 121 exemplos mais curtos
- C7: remove 121 exemplos do meio (comprimento mediano)
- Comparar retention

**Impacto:** Isolamento mais fino do mecanismo — confirma se é puramente quantity ou se position no distribution importa.

### 2.5 Múltiplos datasets

**Pergunta:** O efeito generaliza além de TriviaQA factoid?

**Experimento:** Repetir protocolo com Natural Questions, PopQA, MMLU (como Keisha). Pelo menos 1 dataset adicional com r=16 e r=256.

**Impacto:** Reduz a limitação mais citável do paper atual.

---

## Tier 3 — Exploratório, menor prioridade

### 3.1 Accumulation protocol (G3 do design original)

**Pergunta:** Se acumular dados reais com sintéticos (Gerstgrasser), o threshold muda?

**Experimento:** Rodar r=256 com 10% dados reais + 90% sintéticos. Comparar com replace puro.

**Impacto:** Conecta com Gerstgrasser. Provavelmente funciona, mas menos novel.

### 3.2 Temperature como variável

**Pergunta:** Geração com T=1.0 ou T=1.5 (mais diversa) estabiliza o regime?

**Experimento:** r=256 com T=1.2 em vez de T=0.7. Se diversidade alta previne degradação, fortalece a narrativa de data-quality.

**Impacto:** Modesto — é variação de hyperparameter.

### 3.3 DoRA / IA3 / outros PEFT methods

**Pergunta:** Outros métodos PEFT têm thresholds diferentes?

**Experimento:** Repetir dose-response com DoRA (rank-stabilized), IA3 (scaling only), BitFit (bias only).

**Impacto:** Generalização do achado para a família PEFT.

### 3.4 Continuous pre-training vs instruction fine-tuning

**Pergunta:** O efeito aparece em pre-training continuado (sem chat template)?

**Experimento:** Pegar um base model (sem instruct), fazer recursive training em texto livre. Comparar com instruct.

**Impacto:** Testa se instruction-following formatting é uma variável confundidora.

### 3.5 Real-world synthetic contamination simulation

**Pergunta:** Em um cenário web-scale, onde apenas X% do corpus é sintético e aumenta com o tempo, quando o threshold é cruzado?

**Experimento:** Simular pipeline com proporção crescente de sintético (1%, 5%, 10%, 25%, 50%, 100%) por geração.

**Impacto:** Conecta com o survey original sobre o "paradoxo" — quando a contaminação web cruza o threshold.

### 3.6 Knowledge-type stratification

**Pergunta:** Fatos de diferentes categorias (pessoa, lugar, data, conceito) colapsam em taxas diferentes?

**Dado existente:** K0 tem items categorizáveis. Dá para fazer offline.

**Impacto:** Modesto, mas colorido para o paper — "pessoas colapsam antes de lugares" etc.

### 3.7 Confidence/calibration no regime degradativo

**Pergunta:** O modelo fica overconfident nos erros (Stage B de Keisha)?

**Experimento:** Re-inferir com logprobs salvos para cada item. Comparar confiança em acertos vs erros por regime.

**Impacto:** Conecta com dangerous competence / AI safety.

### 3.8 Normalização cross-architecture via intrinsic dimension

**Pergunta:** Se medir intrinsic dimension dos representations (PCA/MLE) em cada backbone, o threshold normaliza?

**Experimento:** Extrair hidden states, computar intrinsic dim, normalizar eff_rank por intrinsic_dim.

**Impacto:** Se funcionar, seria a "lei universal" que faltou. Requer GPU para embeddings.

---

## Prioridade sugerida para próximo paper

Se for um paper de mitigação/governança:
1. TCE + QLoRA (1.2)
2. Adaptive rank controller (1.4)
3. Step function mapping (1.1)

Se for um paper mecanístico:
1. Subspace overlap (1.3)
2. H-Neurons (2.3)
3. Logit divergence (2.2)

Se for um paper de generalização:
1. Escala 7B+ (1.5)
2. Múltiplos datasets (2.5)
3. Outros PEFT methods (3.3)

---

## Dados já disponíveis para future work (sem rodar nada)

- Synthetic JSONs por geração (Gen0-10) para todos os configs
- K0_results item-level para Gen10 (FFT/QLoRA) e intervenções
- Diversity metrics CSV completo
- Token decomposition por condição
- Fact-overlap cross-seed
- Architecture parameters de 3 backbones
