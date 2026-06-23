# Grill Session 2 — Revisão com Fonte Externa

**Data:** 2026-06-23  
**Origem:** Feedback de modelo externo (Gemini) + análise do pesquisador  
**Objetivo:** Stress-test do design experimental contra vulnerabilidades metodológicas

---

## Decisões Incorporadas

### D16: Posicionamento narrativo — Indicador, não mecanismo causal

**Problema identificado:** O design não pode provar que CKA↓ *causa* Knowledge Collapse. Tentar alegar causalidade é armadilha.

**Decisão:** O claim do artigo é:

> "Representational instability constitutes an observable early indicator of subsequent knowledge collapse, with predictive capacity superior to traditional metrics."

**Não é:**
> "Representational instability causes knowledge collapse."

**Defesa para reviewer:**

> "Our objective is not to demonstrate that representational instability is the causal mechanism of Knowledge Collapse. Our objective is to investigate whether it constitutes an observable and anticipatory indicator of the process described theoretically by Dohmatob, Xu, and Gerstgrasser."

---

### D17: CKA bifurcado — Global + Factual

**Problema identificado:** Se Stage B existe (fluência estável, factualidade cai), a maior parte das ativações nas camadas intermediárias computa fluência/gramática. Mean pooling da sequência inteira dilui o sinal factual.

**Decisão:** Medir dois CKAs:

| Métrica | Extração | Objetivo |
|---|---|---|
| **CKA-Global** | Mean pooling de toda a sequência | Estabilidade representacional geral |
| **CKA-Factual** | Hidden state do último token do prompt (ou últimos 3) | Estabilidade do gargalo de recuperação factual |

**Resultado esperado ideal:**

| Geração | CKA-Global | CKA-Factual |
|---|---|---|
| 0 | 1.00 | 1.00 |
| 1 | 0.99 | 0.94 |
| 2 | 0.98 | 0.87 |
| 3 | 0.97 | 0.80 |

Isso mostraria: "modelo globalmente parecido consigo mesmo, mas região de recuperação factual degradando."

---

### D18: Adapter Health Metrics — Blindagem contra crítica de LoRA artifact

**Problema identificado:** Reviewer pode alegar que CKA↓ é apenas colapso geométrico do adapter (rank collapse) por dados homogêneos, não Knowledge Collapse real.

**Mecanismo da crítica:** Dados sintéticos recursivos têm entropia decrescente → gradientes de menor variância → matriz LoRA A·B degenera para posto efetivo muito menor que rank 16.

**Decisão:** A cada geração, registrar:

1. **Effective Rank:** exp(H(σ)) onde H é entropia dos singular values normalizados de A·B
2. **Spectral Norm:** ||A·B||₂ (maior singular value)
3. **Frobenius Norm:** ||A·B||_F
4. **Top-k Singular Values:** guardar spectrum completo

**Custo computacional:** ~zero (SVD de matrizes 16×hidden_dim é instantâneo).

**Interpretação:**

| Cenário | Effective Rank | CKA | Accuracy | Interpretação |
|---|---|---|---|---|
| A (ideal) | ≈ constante | ↓ | ↓ | Collapse representacional real — adapter saudável |
| B (problemático) | ↓↓↓ | ↓↓↓ | ↓↓↓ | Pode ser rank collapse do adapter |
| C (inesperado) | ↓↓↓ mas G2 também ↓ | — | — | Artifact de LoRA iterativo |

**Defesa via design G1/G2/G3:** Se effective rank cai em G1 mas não em G2 (ambos usam mesmo LoRA, mesmo LR, mesmos epochs), a degradação não é artefato do adapter — é causada pela natureza dos dados.

**Hipótese auxiliar adicionada:**

> H1c: Representational instability is primarily explained by recursive data contamination rather than adapter-induced optimization effects. (Testável: effective rank G1 vs G2.)

> H1d: Recursive training induces measurable degradation in adapter capacity, reflected by decreasing effective rank and singular value concentration. (Monitored, not primary hypothesis.)

---

### D19: Output Logit Divergence — Substituição de "dataset drift"

**Problema identificado:** Medir "drift do dataset" via embeddings das respostas não funciona para factoid QA curto ("Paris" vs "Lyon" são embeddings próximos mas factuamente opostos).

**Decisão:** Em vez de medir drift dos dados textuais, medir drift da distribuição do modelo diretamente:

Para cada prompt fixo do Probe Set:
- Salvar logits completos (ou top-k logits) em cada geração
- Calcular KL divergence / JS divergence entre distribuições de logits consecutivas

**Justificativa:** Isso alinha com o formalismo de Xu (random walk no espaço de parâmetros manifesta-se como drift na distribuição geradora).

**Cadeia mecanística proposta (narrativa, não causal):**

```
Recursive data contamination
    ↓
Output distribution drift (logit divergence)
    ↓
Representational drift (CKA/ESI)
    ↓
Factual accuracy collapse
```

---

### D20: Três níveis de análise — Explícitos no artigo

| Nível | Métricas | O que responde |
|---|---|---|
| **Distribucional** | Logit divergence, accuracy, entropy | O modelo está gerando outputs diferentes? |
| **Representacional** | CKA-Global, CKA-Factual, ESI | As representações internas estão mudando? |
| **Otimização** | Effective rank, spectral norm, Frobenius | O adapter está saudável? |

O artigo opera no nível **representacional**. Monitora o nível de **otimização** como controle. Referencia o nível **mecanístico** (H-Neurons) como future work.

---

### D21: Limitação explícita + Future Work

**Limitação (escrita no artigo):**

> "This work does not attempt to identify the local neural circuits responsible for collapse. CKA and ESI are employed as macroscopic monitoring instruments, not as mechanistic explanations."

**Future Work:**

> "Investigating whether ESI peaks correlate with activation patterns of hallucination-associated neurons (H-Neurons) as described by Gao et al. (2025) would establish a bridge between the representational monitoring approach proposed here and the neuron-level mechanistic account."

---

### D22: H1b adicionada — Dataset entropy → representational drift → factual collapse

**Nova hipótese (auxiliar):**

> H1b: Recursive training induces measurable output distribution drift (logit divergence), which precedes representational instability and subsequent factual degradation.

**Cadeia temporal esperada:**

```
Logit divergence detectável na Gen t
    ↓ (precede?)
CKA/ESI instabilidade na Gen t+1
    ↓ (precede?)
Accuracy collapse na Gen t+2
```

Se essa cadeia se confirmar, é resultado muito forte — demonstra cascata observável.

---

---

### D23: Cenário B é FALHA, não descoberta — Correção crítica

**Erro estratégico identificado:** Na formulação anterior, sugeri que se o effective rank do LoRA colapsasse junto com CKA e accuracy, isso poderia ser "uma descoberta intermediária". Isso é ERRADO.

**Por quê:** O survey é sobre Model Collapse / Knowledge Collapse / Recursive Training — não sobre PEFT pathology. Se o fenômeno depender do achatamento das matrizes LoRA por dados de baixa entropia, ele NÃO generaliza para full fine-tuning ou pré-treinamento iterativo (que são os cenários reais do paper). O artigo viraria "Como o LoRA falha com dados repetitivos" — publicação completamente diferente e menos relevante.

**Condição OBRIGATÓRIA para aceitar H1:**

```
Effective Rank ≈ estável entre gerações (variação < 20% do valor inicial)
E simultaneamente:
CKA ↓ (G1) vs CKA ≈ estável (G2)
Accuracy ↓ (G1) vs Accuracy ≈ estável (G2)
```

**Se Cenário B aparecer no M2 (effective rank desabando):**
- PARAR antes de M3
- Investigar se é artefato de PEFT
- Possíveis ações: aumentar rank (32 ou 64), testar DoRA/IA³, ou considerar head-only fine-tuning

**As métricas espectrais não são análise complementar. São checagem de sanidade obrigatória.**

**Resposta para reviewer (se Cenário A se confirmar):**

> "We explicitly monitored adapter capacity throughout all generations. Effective rank, spectral norm, and Frobenius norm remained statistically stable across groups, indicating that the observed representational instability cannot be explained by low-rank adapter collapse. The effect emerged selectively under recursive synthetic training despite comparable adapter health metrics, confirming its association with knowledge degradation rather than PEFT-specific optimization failure."

---

### D24: Experimento de controle por rank (Nível 2, se necessário)

Se M2 mostrar resultados ambíguos no effective rank:

- Rodar versão reduzida (3 gerações, 1 seed, G1) com rank 8, 16, 32
- Se Lead Time e padrão qualitativo persistem independente do rank → fenômeno não é artifact
- Se desaparece com rank alto → PEFT-specific, pivot necessário

**Custo:** ~1 dia adicional. Decisão tomada apenas após M2.

---

## Vulnerabilidades Não Resolvidas (pendentes de mais informação)

1. **QLoRA + merge iterativo vs full fine-tune:** Nenhum paper estudou collapse neste regime específico. Pode ser mais lento que full fine-tune, ou pode ser indetectável no CKA global. M2 é o checkpoint para decidir.

2. **Comparabilidade com Keisha:** Nosso design é fundamentalmente diferente (QLoRA vs full FT, factoid QA vs WikiText). Não podemos alegar reprodução direta — apenas "investigação independente sob condições controladas."

3. **Suficiência de 10 gerações com QLoRA:** Se o adapter é conservador demais, pode precisar de mais gerações ou configuração mais agressiva. Plano de escape: aumentar LR para 2e-5, aumentar epochs para 3, ou fazer full fine-tune da head layer.

---

---

### D25: Variar rank NÃO resolve Cenário B — Correção ao Nível 2/3

**Crítica recebida:** Se a causa do rank collapse é a baixa entropia dos dados (não a capacidade do adapter), aumentar rank de 16 para 32 ou trocar para DoRA/IA³ é irrelevante. r_eff = f(dados), não f(rank_max).

**Decisão:** Remover Nível 2 e Nível 3 como soluções para Cenário B. Variar rank pode ser útil como caracterização, mas não resolve o problema fundamental.

**Consequência:** Se Cenário B aparecer, a única solução real seria Full Fine-Tuning como controle — que não cabe na RTX 3070 para modelos 3-4B. Isso é uma limitação estrutural do hardware.

---

### D26: NÃO intervir na geração antes de demonstrar o fenômeno

**Crítica recebida:** Sugestão de aumentar temperature (>1.0), relaxar top-p, adicionar repetition penalty para "manter entropia" e preservar o adapter saudável.

**Decisão:** REJEITADA como protocolo do experimento principal.

**Justificativa:** Intervir na geração antes de caracterizar o fenômeno natural é um erro metodológico. Introduz variável confundidora. Pode mascarar exatamente o que se está tentando estudar. O risco é "consertar" antes de entender.

**Protocolo correto:**

1. **M2 (experimento principal):** T=0.7, top_p=0.9, sem penalties. Observar naturalmente o que acontece com entropia, r_eff, CKA, accuracy.
2. **Experimento de mitigação (posterior, se Cenário B aparecer):** Comparar T=0.7 vs T=1.2, top_p 0.9 vs 0.98, com/sem repetition penalty. Pergunta: "A degradação factual persiste quando a entropia do dataset é forçada a permanecer alta?"

**A sequência correta é:**
```
Caracterizar o fenômeno natural → Depois intervir para testá-lo
```

**NÃO:**
```
Intervir preventivamente → Observar fenômeno atenuado
```

---

### D27: Cenário B no M2 não é condição de PARADA — é condição de REFORMULAÇÃO

**Divergência da posição anterior:** Anteriormente documentei que se r_eff cair no M2, o experimento para. Corrijo: o experimento não para. A pergunta se reformula.

**Se M2 mostrar r_eff↓ junto com CKA↓ e accuracy↓:**

A pergunta muda de:
> "CKA/ESI detecta Knowledge Collapse precocemente?"

Para:
> "Qual é a relação entre entropia do dataset sintético, capacidade efetiva do adapter, e degradação factual sob recursive training?"

Isso continua sendo ciência válida. Pode não ser o artigo originalmente planejado, mas é um resultado publicável.

**O que realmente mata o artigo:** NÃO é o Cenário B em si. É se G1 e G2 se comportarem igual no effective rank (indicando que é artefato de fine-tuning iterativo, não de recursão sintética).

**Condição real de falha irrecuperável:**
```
r_eff(G1) ≈ r_eff(G2) ≈ descendo
E
CKA(G1) ≈ CKA(G2) ≈ descendo
E
Accuracy(G1) ≈ Accuracy(G2) ≈ descendo
```

Nesse caso: tudo é forgetting + adapter degeneration. Não há sinal de recursive collapse.

---

### D28: Métricas de entropia do dataset sintético — Obrigatórias no M2

**Decisão:** A cada geração, medir a entropia do dataset sintético gerado:

| Métrica | Sobre o quê |
|---|---|
| Token-level entropy | Distribuição de tokens no corpus sintético gerado |
| Distinct-1/2/3/4 | Diversidade de n-grams |
| TTR (Type-Token Ratio) | Riqueza vocabular |
| Response diversity | Quantas respostas distintas foram geradas (para mesma pergunta, se T>0) |

**Objetivo:** Caracterizar se e quão rápido a "caixa de dados" se homogeniza entre gerações. Isso conecta diretamente com Dohmatob (sampling error perde caudas) e Seddik (convergência para Dirac mass).

**Cadeia observável esperada:**

```
Gen 0: dados reais → alta entropia, alta diversidade
Gen 1: sintéticos → entropia começa a cair
Gen 2: sintéticos de Gen1 → entropia cai mais
...
Gen N: entropia ≈ mínima (quase todas as respostas iguais)
```

Se essa cadeia é simultânea com CKA↓ e accuracy↓, temos evidência de que o colapso distributivo dos dados está driving o colapso representacional. Se CKA cai ANTES da entropia cair significativamente, temos early warning real.

---

## Resumo de Métricas Atualizadas (pós-grill 2 completo)

| Família | Métrica | Implementação |
|---|---|---|
| Accuracy | Exact match normalizado | Evaluation Set (1k) |
| Confiança | Avg log-prob, entropy | Evaluation Set |
| Fluência | External PPL (GPT-2), D4 | Evaluation Set |
| Representação Global | CKA-Global (mean pool) | Probe Set (200), por bloco |
| Representação Factual | CKA-Factual (last token) | Probe Set (200), por bloco |
| Explicabilidade | Attention rollout + ESI | Probe Set (200) |
| Distribucional (modelo) | Logit divergence (KL/JS entre gerações) | Probe Set (200) |
| Distribucional (dados) | Token entropy, Distinct-n, TTR, response diversity | Dataset sintético gerado |
| Otimização | Effective rank, spectral norm, Frobenius, SVD spectrum | Matrizes LoRA A·B |
| Dinâmica | Train loss, grad norm, parameter delta | Por geração |


---

## Conclusão do Grill 2

**Consenso atingido:** O design experimental está completo. Continuar debatendo hipóteses sem dados tem retorno decrescente. Os tensores precisam falar.

**Nível de rigor alcançado:** O protocolo cobre todos os flancos previsíveis de revisão:
- Reviewer ataca dados → métricas de entropia do dataset
- Reviewer ataca adapter → espectro do LoRA (effective rank)
- Reviewer ataca diluição do sinal → CKA-Factual (last token)
- Reviewer ataca forgetting → divergência G1 vs G2 vs G3
- Reviewer ataca causalidade → claim é "indicador", não "causa"
- Reviewer ataca attention → CKA como validação independente

---

## Decisões Finais de Implementação (M1A)

### D29: Ordem de implementação do M1A

A prioridade NÃO é treinar. É validar que a instrumentação funciona.

**Ordem:**

1. `extract_representations` — extrair hidden states, attentions, logits de um prompt
2. `cka` — CKA(modelo, modelo) ≈ 1.0 (sanity check)
3. `attention_rollout` — ESI(modelo, modelo) ≈ 0.0 (sanity check)
4. `lora_spectrum` — SVD de matrizes LoRA (para quando existir adapter)
5. Só depois: generation pipeline + QLoRA training

**Pergunta que M1A responde:**

> "Consigo observar de forma confiável os sinais que pretendo transformar em evidência científica?"

### D30: CKA-Factual — Implementação

Extrair hidden state APENAS do último token do prompt (position -1 antes da geração).

```python
last_token_hidden = hidden_states[layer][:, -1, :]  # (1, hidden_dim)
```

Para o Probe Set (200 amostras): empilhar → (200, hidden_dim) por camada.

Comparar entre gerações usando CKA linear.

### D31: Logit divergence — Implementação

Para cada prompt do Probe Set:
- Salvar logits do primeiro token gerado (posição de resposta, último do prompt)
- Normalizar para distribuição (softmax)
- Calcular JS divergence entre Gen t e Gen t+1

Agregar: média sobre 200 amostras.

### D32: Métricas de entropia do dataset sintético

A cada geração, sobre o corpus de respostas geradas pelo modelo:
- Token-level entropy (distribuição de tokens)
- Distinct-1/2/3/4
- TTR (Type-Token Ratio)
- Response diversity (respostas únicas / total de respostas)

---

## Status Final

O design teórico está **fechado**. Próximo passo: implementação do M1A atualizado com as novas métricas (CKA-Factual, logit divergence, adapter health placeholder).

---

## Decisões de Engenharia (Implementação M1A)

### D33: Extração via Forward Hooks com redução imediata — NÃO retornar tensores brutos

**Problema identificado:** `output_hidden_states=True` + `output_attentions=True` em modelo 3-4B com 32 camadas e seq_len ~512 estoura 8GB VRAM imediatamente. Attentions são [B, H, L, L] × 32 camadas.

**Decisão:** Usar forward hooks do PyTorch que:
1. Capturam output de cada camada durante forward pass
2. Reduzem IMEDIATAMENTE para as representações de interesse (mean_pooled, last_token)
3. Movem para CPU (.cpu())
4. Descartam o tensor original

**Arquitetura:**

```python
class RepresentationProbe:
    def _get_hook(self, name):
        def hook(module, input, output):
            tensor = output[0].detach() if isinstance(output, tuple) else output.detach()
            self.features[name] = {
                "global": tensor.mean(dim=1).cpu(),      # mean pool → (1, hidden_dim)
                "factual": tensor[:, prompt_end_idx, :].cpu()  # last prompt token
            }
        return hook
```

**Benefício:** Footprint de memória cai de ~GB (todos os hidden states + attentions brutos) para ~MB (apenas vetores reduzidos na CPU).

---

### D34: Token index do CKA-Factual — NÃO usar tensor[:, -1, :]

**Problema:** `[:, -1, :]` é o último token da sequência processada, que pode não ser o último token do prompt (pode ser padding, EOS, ou já um token gerado).

**Decisão:** Calcular explicitamente `prompt_end_idx` baseado no comprimento do input tokenizado:

```python
prompt_ids = tokenizer(prompt, return_tensors="pt")
prompt_end_idx = prompt_ids.input_ids.shape[1] - 1  # último token real do prompt
```

Guardar esse índice e usá-lo no hook para extrair o hidden state correto.

---

### D35: Teste de sensibilidade do CKA — Verificar que não é insensível

**Problema:** Se CKA(M, M') ≈ 1.0 mesmo após mudanças relevantes, a métrica é pouco informativa e não vai detectar nada.

**Decisão:** No M1A, além do sanity check CKA(M,M) ≈ 1.0, fazer um teste de sensibilidade:
- Adicionar perturbação gaussiana pequena aos hidden states (σ = 0.01, 0.05, 0.1)
- Verificar que CKA decresce monotonicamente com a perturbação
- Isso confirma que a métrica tem resolução suficiente para detectar drift real

**Se CKA for insensível mesmo com perturbação moderada:** considerar métricas alternativas (Procrustes distance, PWCCA) ou ajustar granularidade (por neurônio em vez de por camada).

---

### D36: Atenção extraída separadamente — NÃO no mesmo forward pass que hidden states

**Decisão para modelo 3-4B em 8GB:**
- Forward pass 1: extrair hidden states via hooks (global + factual) — `output_attentions=False`
- Forward pass 2: extrair attention maps via hooks — `output_hidden_states=False`
- Isso duplica o tempo mas evita OOM

**Para modelo piloto (1.5B):** pode fazer ambos no mesmo pass (cabe na memória).

**Trade-off aceito:** 2× mais lento no probe set (200 prompts × 2 passes = ~10min extra). Aceitável.

---

### D37: NÃO salvar Parquet dentro do hook — Buffer + flush

**Problema:** Escrever em disco dentro do forward hook cria gargalo de I/O que trava a GPU.

**Decisão:** Hook apenas faz: detach → reduce → .cpu() → append ao buffer in-memory. Flush para disco acontece fora do forward, a cada N amostras (ex: 50 ou 100).

```python
# Dentro do hook:
self.buffer[name].append({"global": mean_pooled, "factual": last_token})

# Fora do loop de inferência:
if len(self.buffer[name]) >= flush_interval:
    self._flush_to_disk(name)
```

---

### D38: prompt_end_idx explícito — Calculado na tokenização, não inferido de attention_mask

**Problema:** Modelos instruct (Gemma, Qwen) usam chat templates com tokens especiais (`<system>`, `<user>`, `<assistant>`). O `attention_mask.sum() - 1` pode apontar para um token de template, não para o último token semântico do prompt.

**Decisão:** Calcular `prompt_end_idx` durante a tokenização e passá-lo explicitamente para o hook:

```python
# Na tokenização:
full_input = tokenizer.apply_chat_template(messages, ...)
# O último token ANTES de "<assistant>" é o prompt_end_idx
prompt_end_idx = find_assistant_start(full_input) - 1
```

Armazenar como `self.current_prompt_end_indices` na classe Probe, acessível pelo hook.

---

### D39: CKA-Factual com janela de tokens (1, 3, 5) — Não apenas last token

**Decisão:** Extrair hidden states dos últimos 1, 3 e 5 tokens do prompt (antes do ponto de geração). Custo adicional desprezível.

```python
factual_1 = tensor[i, prompt_end_idx, :]
factual_3 = tensor[i, prompt_end_idx-2:prompt_end_idx+1, :].mean(dim=0)
factual_5 = tensor[i, prompt_end_idx-4:prompt_end_idx+1, :].mean(dim=0)
```

No M2, comparar sensibilidade de CKA-1 vs CKA-3 vs CKA-5. Usar a mais discriminativa no M3.

---

### D40: Teste de sensibilidade — Curva completa com 4 níveis

**Protocolo no M1A:**

| Teste | Comparação | CKA esperado |
|---|---|---|
| A (identidade) | M vs M (mesma seed) | 1.0000 |
| B (ruído mínimo) | M vs M + N(0, 1e-5) nos pesos de atenção | ~0.999+ |
| C (ruído moderado) | M vs M + N(0, 1e-4) | ~0.99 |
| D (ruído significativo) | M vs M + N(0, 1e-3) | ~0.95? |

Se a curva NÃO for monotonicamente decrescente, CKA está bugado ou insensível. Não prosseguir.

---

### D41: Probe Set estratificado por categoria de conhecimento

**Decisão:** Montar o Probe Set (200 amostras) com distribuição intencional:

| Categoria | Exemplos | N |
|---|---|---|
| Pessoa | "Who wrote Hamlet?" | 50 |
| Local | "What is the capital of Japan?" | 50 |
| Data/Número | "What year did WWII end?" | 50 |
| Conceito/Ciência | "What is the chemical symbol for gold?" | 50 |

**Benefício:** Permite análise pós-hoc: "O colapso aparece primeiro em qual categoria de conhecimento?" Resultado secundário potencialmente publicável.

---

## Status Final (definitivo)

O design teórico E de engenharia está **fechado**. Não há mais decisões pendentes.

**M1A v1 — Escopo final:**
- Hooks com redução imediata + buffer + flush
- CKA Global (mean pool)
- CKA Factual (last 1/3/5 tokens do prompt)
- Attention Rollout + ESI
- Effective Rank + Spectral Norm + Frobenius Norm (placeholder para pós-LoRA)
- Teste de identidade (CKA ≈ 1, ESI ≈ 0)
- Teste de sensibilidade (curva de perturbação gaussiana)
- Accuracy (exact match)
- Confidence (log-prob, entropy)
- Probe Set estratificado (4 categorias × 50)

**Próximo gargalo:** empírico, não teórico. Rodar M1A.

---

### D42: CKA-raw vs CKA-normalized — Medir ambos

**Problema:** LoRA pode alterar norma/escala das ativações sem mudar orientação geométrica. Se CKA opera sobre representações brutas, mudança de magnitude pode ser confundida com mudança de representação.

**Decisão:** Calcular dois CKAs:
- **CKA-raw:** sobre hidden states brutos (como implementado no linear_cka atual — já centra por mean, mas não normaliza escala)
- **CKA-normalized:** sobre hidden states z-score normalizados por dimensão antes do cálculo

Se ambos contam a mesma história: validação cruzada. Se divergem: achado importante (magnitude vs geometria).

Custo: praticamente zero (normalização é O(n)).

---

### D43: Teste de sensibilidade inclui checkpoint real (não apenas ruído gaussiano)

**Problema:** Perturbação gaussiana nos pesos é artificial. Pode produzir resposta diferente da que o treinamento real produz. CKA pode ser sensível a ruído mas insensível a transformações de gradiente.

**Decisão:** Adicionar ao protocolo de sensibilidade:

| Teste | Comparação | O que valida |
|---|---|---|
| A | M vs M (identidade) | Instrumentação correta |
| B | M vs M + N(0, 1e-5) | Sensibilidade mínima |
| C | M vs M + N(0, 1e-4) | Sensibilidade moderada |
| D | M vs M pós-1-mini-step (dados reais) | Sensibilidade a transformação REAL |

Se A≈1, B<1, C<B, D<C → métrica calibrada e sensível a mudanças reais.
Se A≈1, B<1, C<B, D≈A → CKA insensível a training real. Problemático.

---

## DESIGN CONGELADO — NÃO ADICIONAR MAIS NADA

**Data de congelamento definitivo:** 2026-06-23T01:16

A partir deste ponto, qualquer decisão adicional será tomada com base em DADOS do M1A/M2, não em especulação teórica.

**Lista fechada do M1A v1:**

1. Forward hooks com redução imediata (GPU → CPU)
2. Buffer in-memory + flush para disco a cada N amostras
3. prompt_end_idx calculado na tokenização (chat template aware)
4. CKA Global (mean pool de tokens do prompt)
5. CKA Factual (últimos 1/3/5 tokens, extraídos como slice [-5:])
6. CKA-raw + CKA-normalized
7. Attention rollout + ESI
8. Effective Rank + Spectral Norm + Frobenius Norm (placeholder)
9. Accuracy (exact match normalizado)
10. Confidence (avg log-prob + predictive entropy)
11. Calibração: identidade, perturbação (1e-5, 1e-4), 1-mini-step

**O que NÃO entra no M1A:**
- H-Neurons, TruthfulQA, dataset drift, temperatura adaptativa, mitigações, novos PEFTs, FFT, logit divergence completa (isso entra no M1B quando já houver 2 gerações para comparar)

---

### D44: RETIRADA — CKA-normalized NÃO será implementado

**Correção:** A decisão D42 (CKA-raw + CKA-normalized) está parcialmente ERRADA. Remover CKA-normalized.

**Motivo:** Z-score por dimensão destrói outlier dimensions que LLMs usam como mecanismos de roteamento. Esmaga sinal semântico real e amplifica ruído em dimensões irrelevantes. Escalonamento anisotrópico causado pelo LoRA É mudança representacional válida — deve ser capturado pelo CKA, não normalizado.

Adicionalmente: CKA Linear já é invariante a escala isotrópica (a centralização dupla H remove translação). Normalizar antes é redundante na melhor das hipóteses e destrutivo na pior.

**Decisão final:** Usar APENAS hidden states originais. Sem Z-score. Sem normalização pré-CKA.

---

### D45: Cast obrigatório para float32 antes de qualquer métrica

**Problema:** Ativações saem em bf16/fp16 (QLoRA). Cálculos de CKA envolvem X^T @ X em matrizes [200, 3072+]. FP16 estoura (max ~65500) → NaN/Inf silenciosos.

**Regra absoluta:**

```python
# Dentro do hook, após offload:
tensor_cpu = tensor.cpu().float()  # SEMPRE float32 para métricas
```

| Contexto | Precisão |
|---|---|
| Inferência do modelo | bf16/fp16 (normal) |
| Cálculo de CKA | float32 obrigatório |
| Cálculo de SVD (effective rank) | float32 obrigatório |
| Cálculo de normas | float32 obrigatório |
| Attention rollout | float32 obrigatório |

---

## M1A v1 — ESPECIFICAÇÃO FINAL CONGELADA

**Data:** 2026-06-23T01:17

Não há mais decisões de design pendentes. Implementar exatamente isto:

1. Forward hooks nas camadas selecionadas (early/middle/late)
2. Redução imediata na GPU: mean pool (Global) + slice [-5:] (Factual)
3. CPU offload com cast para float32
4. Buffer in-memory + flush para disco (Parquet) a cada N amostras
5. prompt_end_idx pré-calculado na tokenização (chat template aware)
6. CKA Global (mean pool dos tokens do prompt, sem normalização)
7. CKA Factual-1, Factual-3, Factual-5 (média dos últimos 1/3/5 tokens)
8. Attention rollout + ESI (segundo forward pass para modelo 3-4B)
9. Effective Rank + Spectral Norm + Frobenius Norm (placeholder para pós-LoRA)
10. Accuracy (exact match) + Confidence (log-prob, entropy)
11. Calibração: identidade (≈1.0), ruído 1e-5, ruído 1e-4, 1-mini-step real
12. Probe Set: 200 prompts estratificados (implementar estratificação no M1B, M1A pode usar qualquer 200)

---

### D46: M0 obrigatório — Introspecção do modelo antes de escrever extractor

**Problema:** O caminho dos módulos varia por arquitetura (Gemma: `model.model.layers` vs `model.language_model.model.layers`, Qwen: `model.model.layers`). Hooks no módulo errado = dados inválidos silenciosamente.

**Decisão:** Antes de implementar qualquer hook, executar:

```python
model = load_model(...)
for name, module in model.named_modules():
    print(name, type(module))
```

E mapear:
- Caminho exato para layers (decoder blocks)
- Número real de camadas
- Hidden dim real
- Qual output format (tuple vs tensor)
- VRAM consumida após carregamento

**Isso define:** quais índices usar para early/middle/late, como acessar os módulos no hook, se cabe na memória com ambos output_hidden_states e output_attentions.

---

### D47: Edge cases no Factual-3/5 — Proteger contra prompts curtos

```python
start_f3 = max(0, end_idx - 2)
start_f5 = max(0, end_idx - 4)
f3 = tensor[i, start_f3:end_idx+1, :].mean(dim=0)  # pad-safe
f5 = tensor[i, start_f5:end_idx+1, :].mean(dim=0)
```

---

### D48: Buffer com flush — Implementar desde o início

```python
MAX_BUFFER_SIZE = 50  # flush a cada 50 amostras

def _maybe_flush(self, layer_idx):
    if len(self.buffer[layer_idx]["global"]) >= MAX_BUFFER_SIZE:
        self._flush_to_disk(layer_idx)
```

---

### D49: Teste de identidade — NÃO usar assert. Registrar e analisar.

Não travar o pipeline com `assert np.isclose(cka, 1.0, atol=1e-4)`. Variações numéricas legítimas (CPU, BLAS, ordem de operações) podem gerar 0.9997. Registrar o valor e analisar post-hoc.

---

### D50: Sequência final de implementação

| Step | Nome | Objetivo | Dependência |
|---|---|---|---|
| **M0** | Model Introspection | Carregar modelo, mapear módulos, medir VRAM | Nenhuma |
| **M1A-a** | Representation Extraction | Hooks + factual tokens + offload + buffer | M0 |
| **M1A-b** | Metric Validation | CKA(M,M), CKA(M,M+ε), effective rank | M1A-a |
| **M1A-c** | Accuracy + Confidence | Exact match, log-prob, entropy | M0 |
| **M1B** | Single Cycle | QLoRA → merge → re-evaluate | M1A completo |
| **M2** | 3 Generations | Threshold GFW, adapter health, sinal | M1B |

**REGRA:** Não avançar para step N+1 até step N funcionar.

