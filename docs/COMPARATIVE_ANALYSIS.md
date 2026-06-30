# Análise Comparativa: Lições dos Artigos Base para Melhorar o Paper

**Objetivo:** Identificar lições de escrita, claims, venda, impacto, precisão e memorabilidade dos artigos base que podem melhorar nosso manuscrito.

**Paper alvo:** "Effective Training Pressure Gates Recursive Knowledge Degradation in LLMs"

---

## Item 1: Shumailov et al. (2024) — Nature

*"AI models collapse when trained on recursively generated data"*
*Nature, Vol. 631, pp. 755-759*

### Escrita

**O que fazem bem:**
- O abstract abre com uma frase cultural/contextual ("Stable Diffusion revolutionised image creation...") antes de entrar no problema técnico. Isso ancora o leitor não-especialista.
- A pergunta central é formulada como uma questão geracional: "What will happen to GPT-{n}?" Extremamente memorável.
- Usam uma única frase para o resultado principal: "use of model-generated content in training causes irreversible defects in the resulting models, where tails of the original content distribution disappear."
- O título é uma afirmação direta e definitiva: "makes models forget" (verbo forte, sem hedging).

**Lição para nosso paper:**
- Nossa abertura é funcional mas não memorável. Poderíamos ter uma frase-gancho equivalente a "What will happen to GPT-{n}?" — algo como "How many recursive generations can a QLoRA adapter survive?"
- O título deles é mais acessível. O nosso ("Effective Training Pressure Gates Recursive Knowledge Degradation") é técnico e menos citável em conversas. Considerar alternativa mais direta.

### Claims

**O que fazem:**
- Claim central absoluto: "irreversible defects." Sem hedging.
- Suporte: demonstração em 3 classes de modelos (GMM, VAE, LLM).
- Generalização: "ubiquity amongst all learned generative models."

**Lição para nosso paper:**
- Nós somos mais conservadores (correto para nosso escopo), mas talvez conservadores demais. A frase "is not an inevitable consequence" poderia ser reescrita como "is controllable" — mais direta, igualmente verdadeira.
- Shumailov faz claims fortes porque tem teoria + empiria + 3 modelos. Nós temos 3 backbones + 3 axes. Podemos ser mais assertivos na Conclusion.

### Venda / Impacto

**O que fazem:**
- Vendem o paper como um **alerta para o futuro da internet**. Não como um resultado técnico.
- A implicação prática é óbvia: "the value of data collected about genuine human interactions will be increasingly valuable."
- Publicaram na **Nature** (não em venue de ML) — estratégia de impacto interdisciplinar.

**Lição para nosso paper:**
- Nós vendemos como "engineering problem." Está correto para EAAI. Mas poderíamos ser mais enfáticos na implicação: "practitioners currently have no principled way to determine safe adapter configurations for recursive pipelines." Essa frase cria urgência.
- A frase final da Conclusion poderia ecoar o estilo deles: em vez de terminar com "remains an open challenge", terminar com uma implicação prática memorável.

### Precisão

**O que fazem:**
- Precisão teórica forte (Wasserstein bounds, provas formais).
- Mas os experimentos empíricos em LLMs são relativamente simples (OPT-125M).
- Não fazem dose-response. Não mapeiam thresholds. Não testam intervenções.

**Lição para nosso paper:**
- Nosso paper é empiricamente muito mais rico que Shumailov. 6 ranks, 4 LRs, 5 intervenções, 3 backbones, 10 gerações, N=3. Isso é uma vantagem que podemos vender melhor.
- Podemos dizer explicitamente: "Prior empirical demonstrations of collapse used single configurations; we provide the first systematic dose-response characterization."

### Memorabilidade

**O que faz esse paper ser lembrado:**
1. O nome "model collapse" (eles o cunharam).
2. A analogia com "GAN mode collapse" (familiar ao público).
3. A pergunta "What happens to GPT-{n}?"
4. O resultado "tails disappear" (visual, intuitivo).
5. A publicação na Nature (prestígio + visibilidade).

**Lição para nosso paper:**
- Nosso conceito memorável é "effective training pressure" + "pressure-gated." Precisamos que esse termo se fixe.
- O resultado mais visual/memorável que temos é o **r=128 dissociation** (retention ok, distribution collapsed). Isso é contraintuitivo e surpreendente. Deveria estar no abstract mais prominentemente.
- Também temos: "~5% reduction flips the regime." Esse número é memorável. Deveria ser destacado.

### Resumo de ações concretas para o paper

| Dimensão | Ação sugerida |
|---|---|
| Escrita | Considerar um hook mais memorável na abertura da Intro |
| Claims | Podemos ser ligeiramente mais assertivos na Conclusion |
| Venda | Enfatizar que nenhum framework existia antes para decisão de rank em pipelines recursivos |
| Impacto | Destacar superioridade empírica sobre Shumailov (que usou OPT-125M single config) |
| Precisão | Já somos mais precisos empiricamente; mencionar isso no posicionamento |
| Memorabilidade | Garantir que "pressure-gated", "r=128 dissociation" e "5%" sejam os 3 takeaways |

---

## Item 2: Dohmatob et al. (2025) — ICLR

*(próximo item — aguardando)*


## Item 2: Dohmatob et al. (2025) — ICLR

*"Strong Model Collapse"*
*ICLR 2025, Meta FAIR*

### Escrita

**O que fazem bem:**
- O abstract abre dentro do paradigma de "scaling laws" — posiciona o paper imediatamente no debate mais quente de ML.
- A frase-chave é cirúrgica: "even the smallest fraction of synthetic data (e.g., as little as 1%) can still lead to model collapse: larger and larger training sets do not enhance performance."
- Usam uma progressão teoria → empiria → implicação. Cada seção constrói sobre a anterior.
- O título "Strong Model Collapse" é uma extensão deliberada do "Model Collapse" de Shumailov. Posiciona-se como sequência natural.

**Lição para nosso paper:**
- A abertura deles ancora no paradigma dominante (scaling laws). Nosso paper poderia ancorar mais fortemente no paradigma de "PEFT como padrão de facto."
- A frase "as little as 1%" cria shock value. Nosso equivalente seria "as little as 5% reduction restores stability" — já temos, mas está enterrado nos Results.

### Claims

**O que fazem:**
- "Strong" model collapse: mais forte que Shumailov (inevitável mesmo com 1%).
- "Larger models amplify collapse" — resultado contraintuitivo que gera citações.
- Suporte: provas formais (Theorem 1), scaling laws regime, random projections.
- Claim negativo poderoso: "cannot be fixed by naively mixing synthetic and real data."

**Lição para nosso paper:**
- Dohmatob faz claims sobre o que NÃO funciona (mixing não resolve). Nós fazemos claims sobre o que FUNCIONA (exposure reduction resolve no boundary). Essa é uma vantagem narrativa que devemos explorar mais.
- O resultado contraintuitivo deles é "bigger = worse." O nosso é "r=128 looks stable but isn't" (dissociation). Ambos geram surprise. Devemos vender melhor a surpresa.

### Venda / Impacto

**O que fazem:**
- Vendem como extensão do debate scaling laws.
- Vendem como resultado negativo universal: nenhuma quantidade de dados salva.
- Meta FAIR como afiliação adiciona peso institucional.

**Lição para nosso paper:**
- Posicionar como a **resposta prática** ao pessimismo de Dohmatob: "They showed collapse is inevitable under their assumptions. We show those assumptions don't hold under low-rank PEFT, except when pressure exceeds the threshold."
- Esse posicionamento de "resposta a um pessimismo teórico" é muito citável.

### Precisão

**O que fazem:**
- Extremamente rigoroso matematicamente (operator-valued free probability theory).
- Mas os experimentos empíricos são em regressão linear e random features, não em LLMs full-scale.

**Lição para nosso paper:**
- Somos o oposto: empiricamente ricos, teoricamente leves. Complementar.
- Podemos dizer: "The bounds of Dohmatob et al. predict linear error growth. Our observations are consistent in high-pressure regimes, while low-pressure PEFT configurations fall outside their model's assumptions."

### Memorabilidade

**O que faz esse paper ser lembrado:**
1. "Even 1% synthetic data causes collapse" — número chocante.
2. "Bigger models amplify collapse" — contraintuitivo.
3. "Strong" (extensão de Shumailov).
4. Conexão com scaling laws.
5. ICLR 2025 + Meta FAIR.

**Lição para nosso paper:**
- Nossos "números chocantes": "5% flips regime", "10× threshold difference", "88.6% retention but 0.13 efficiency."
- Garantir que pelo menos 2 apareçam no abstract/highlights.

### Resumo de ações concretas

| Dimensão | Ação sugerida |
|---|---|
| Escrita | Ancorar abertura no paradigma PEFT dominante |
| Claims | "Resposta prática ao pessimismo teórico" |
| Venda | Contraste: "inevitable under theirs; controllable under ours" |
| Impacto | Seus bounds valem no regime de alta pressão, não no de baixa |
| Precisão | Complementar: empiria rica vs teoria deles |
| Memorabilidade | Destacar 3 números chocantes |

---

## Item 3: Keisha et al. (2025) — Knowledge Collapse

*(próximo item — aguardando)*


## Item 3: Keisha et al. (2025) — Knowledge Collapse

*"Knowledge Collapse in LLMs: When Fluency Survives but Facts Fail under Recursive Synthetic Training"*
*arXiv:2509.04796, UCL / Holistic AI*

### Escrita

**O que fazem bem:**
- O subtítulo "When Fluency Survives but Facts Fail" é extremamente memorável. Resume o paper inteiro em 6 palavras.
- As 4 contribuições são listadas de forma limpa no final da Intro (numbered, com verbos diferentes).
- A taxonomia Stage A/B/C cria um vocabulário compartilhado que outros papers podem usar.
- "Confidently wrong" como conceito é imediatamente compreensível e assustador.

**O que fazem mal:**
- A escrita é competente mas não excepcional. Frases longas, alguns parágrafos densos demais.
- A Introduction mistura Related Work (não há seção separada). Isso funciona em preprint curto mas não em journal.
- Metodologia pouco detalhada (MMLU, synthetic ratios, mas poucos detalhes de training).

**Lição para nosso paper:**
- O subtítulo deles é melhor que o nosso título inteiro como "soundbite." Considerar se "Effective Training Pressure" precisa de um subtítulo mais acessível.
- A taxonomia Stage A/B/C é poderosa. Nossa taxonomia (homeostatic/bounded/degradative) é igualmente boa mas menos catchy. "Bounded" não comunica tanto quanto "confidently wrong."

### Claims

**O que fazem:**
- 4 contribuições claras: (1) define knowledge collapse, (2) mostra sensibilidade a formato, (3) propõe domain-specific mitigation (15× improvement), (4) framework de avaliação.
- Estatística formal: F(4, 1960)=5.92, p<10^-3 para sensibilidade; p<0.001 para mitigação.
- Claim bold: "15× improvement in collapse resistance."

**O que fazem mal:**
- Single model (Gemma 3 1B IT). Single dataset family (MMLU). Single method (FFT).
- N=? Não fica claro quantos seeds usam.
- O claim de "15×" é sobre decay rate, não sobre resultado absoluto. Pode confundir.

**Lição para nosso paper:**
- Nós temos 3 backbones, N=3 seeds, 6 ranks, 4 LRs. Empiricamente muito mais sólido.
- Mas eles têm números mais impressionantes na apresentação ("15×"). Nosso equivalente seria: "regime transition spans >16 percentage points" ou "threshold differs by 10× across backbones." Precisamos apresentar nossos números com o mesmo impacto.
- A estatística formal deles (ANOVA F-test) é algo que não temos (N=3 insuficiente). Isso é uma limitação honesta que já declaramos.

### Venda / Impacto

**O que fazem:**
- Vendem como problema de **segurança**: "confidently wrong outputs that pose critical risks in accuracy-dependent domains."
- Referências a healthcare (40% error rates). Isso cria urgência imediata.
- Propõem uma solução (domain-specific training). O paper tem "problema + solução."

**O que fazem mal:**
- A solução proposta (domain-specific) é bastante limitada e ad-hoc.
- Não explicam por que a solução funciona mecanisticamente.

**Lição para nosso paper:**
- Eles vendem segurança/risco. Nós vendemos engenharia/controle. Ambos são válidos para venues diferentes.
- Para EAAI, nosso framing de "governança de pipelines" é melhor que o framing de "safety risk."
- MAS: poderíamos adicionar uma frase na Intro mencionando o risco prático de outputs "confidently wrong" que nosso r=128 dissociation ilustra. Conecta com audiência de safety sem mudar o foco.

### Precisão

**O que fazem:**
- Usam Gemma 3 1B IT — mesmo backbone que nós!
- Protocolo: FFT recursivo com synthetic ratios (25%, 50%, 75%, 100%).
- Avaliação: MMLU (multiple choice), greedy rate, entropy, perplexity.
- 15 gerações recursivas.

**O que fazem mal:**
- Não fazem dose-response de capacidade (rank/params). Testam ratios de dados.
- Não isolam perturbation magnitude.
- Não testam PEFT/LoRA.
- Single seed aparentemente.

**Lição para nosso paper:**
- A comparação direta é forte: mesmo backbone (Gemma 3 1B), eles observam collapse com FFT; nós observamos homeostase com QLoRA r=4. Isso é exatamente o que nosso Related Work diz ("suggestive context, different protocol").
- Devemos ser MUITO cuidadosos para não parecer que estamos contradizendo Keisha. A frase "different protocol precludes direct comparison" é essencial e já temos.

### Memorabilidade

**O que faz esse paper ser lembrado:**
1. "Confidently wrong" — frase que gruda.
2. Stage A/B/C — taxonomia citável.
3. "Fluency survives but facts fail" — subtítulo perfeito.
4. "15× improvement" — número impressionante.
5. Healthcare connection — cria urgência.

**Lição para nosso paper:**
- Nossos equivalentes memoráveis:
  - "Pressure-gated" (conceito novo)
  - "r=128: retention bounded, distribution collapsed" (surprise)
  - "5% reduction flips the regime" (actionable)
  - "Homeostatic / Bounded / Degradative" (taxonomia)
- Precisamos de uma frase-soundbite equivalente a "fluency survives but facts fail." Candidata: **"Retention survives but quality collapses"** ou **"The model looks stable but isn't."**

### Resumo de ações concretas

| Dimensão | Ação sugerida |
|---|---|
| Escrita | Considerar subtítulo mais acessível ou frase-soundbite no abstract |
| Claims | Apresentar nossos números com mais impacto ("16pp span", "10× threshold") |
| Venda | Mencionar risco de "looks stable but isn't" (r=128) como implicação de safety |
| Impacto | Somos empiricamente superiores (3 backbones, N=3, dose-response). Usar isso. |
| Precisão | Mesmo backbone (Gemma 3). Não contradizer, complementar. |
| Memorabilidade | Precisamos de 1 frase-soundbite equivalente a "fluency survives but facts fail" |

---

## Item 4: Gerstgrasser et al. (2024) — Accumulation

*(próximo item — aguardando)*


## Item 4: Gerstgrasser et al. (2024) — Accumulation

*"Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data"*
*arXiv:2404.01413, Stanford*

### Escrita

**O que fazem bem:**
- O título é uma **pergunta**. "Is Model Collapse Inevitable?" Isso engaja imediatamente. O leitor quer a resposta.
- A resposta está no subtítulo: "Breaking the Curse of Recursion by Accumulating." Completo em si.
- O abstract tem estrutura perfeita: (1) problema, (2) assunção incorreta da literatura, (3) correção, (4) empiria, (5) teoria.
- A Figure 1 resume o paper inteiro visualmente (replace vs accumulate diagrama). Extremamente eficiente.
- Usam "arguably more realistic assumption" — suave mas devastador para a literatura anterior.

**O que fazem mal:**
- O paper é longo (grande número de autores Stanford, muitos experimentos + teoria). Para nosso formato EAAI isso não é um problema, mas mostra que eles gastam espaço.

**Lição para nosso paper:**
- Um título como pergunta ("When Does Recursive Degradation Emerge Under PEFT?") seria memorável. Nosso título é uma afirmação técnica. É preciso mas não convida.
- A frase "arguably more realistic assumption" é uma técnica poderosa: invalida gentilmente a premissa da literatura sem ser agressivo. Nós podemos usar algo similar: "while prior work assumes full fine-tuning, practical adaptation pipelines overwhelmingly use PEFT."
- A Figure 1 tipo diagrama "Replace vs Accumulate" é o que nossa Fig 7 (methodology) tenta fazer. Mas a deles é mais simples e mais impactante.

### Claims

**O que fazem:**
- Claim principal: "accumulating data avoids model collapse." Simples. Limpo.
- Suporte: LLMs (pretraining sequences), diffusion models, VAEs, + prova teórica (linear models).
- A prova é que com acumulação, test error tem upper bound finito (não diverge).
- Contraste direto com Shumailov: under replace → diverge; under accumulate → bounded.

**Lição para nosso paper:**
- A estrutura "under X → bad; under Y → good" é muito clara. Nosso equivalente: "under high pressure → degradation; under low pressure → homeostasis." Já fazemos isso, mas podemos ser mais diretos no abstract/highlights.
- Eles provam um bound teórico. Nós não. Mas temos something melhor para EAAI: a demonstração de que a transição é **sharp** e **manipulável**. Isso é mais actionable.

### Venda / Impacto

**O que fazem:**
- Vendem como "correção de uma premissa da literatura." A premissa replace-only era artificial.
- O paper basicamente diz: "O alarme de Shumailov era sobre um cenário que não corresponde à realidade."
- Impacto: Stanford, 10+ autores (incluindo Donoho, Koyejo — nomes pesados).

**Lição para nosso paper:**
- Nós também "corrigimos uma premissa": a premissa de que FFT é o paradigma relevante. Na prática, PEFT domina. E sob PEFT, o regime de collapse é diferente.
- Mas cuidado: Gerstgrasser opera no eixo **dados** (accumulate vs replace). Nós operamos no eixo **updates** (rank, LR, exposure). São ortogonais. Isso já está claro no Related Work.

### Precisão

**O que fazem:**
- LLMs reais (sequences of GPT-2 scale models pretrained from scratch). Não fine-tuning.
- Diffusion models (molecular conformation). VAEs (images).
- Prova teórica (linear models, bounded MSE under accumulation).
- Múltiplos tamanhos de modelo, arquiteturas, hyperparameters.

**O que fazem mal:**
- Não testam PEFT. Não testam fine-tuning recursivo. O cenário deles é pretraining.
- Não fazem dose-response (não variam a proporção gradualmente).

**Lição para nosso paper:**
- A diferença fundamental: eles fazem pretraining from scratch a cada geração; nós fazemos fine-tuning. São problemas diferentes que podem ter dinâmicas diferentes.
- Nós estamos mais próximos da prática industrial (fine-tuning recursivo com adapters). Isso é uma vantagem para EAAI que devemos enfatizar.

### Memorabilidade

**O que faz esse paper ser lembrado:**
1. Título como pergunta: "Is Model Collapse Inevitable?"
2. A resposta: "No, if you accumulate."
3. Figure 1: diagrama Replace vs Accumulate (visual instantâneo).
4. Upper bound teórico (prova que não diverge).
5. Stanford + Donoho (autoridade).

**Lição para nosso paper:**
- Nosso equivalente de "No, if you accumulate" seria: **"No, if you control pressure."** Essa frase deveria estar no paper.
- O Figure 1 deles funciona porque mostra o contraste em 1 imagem. Nosso Fig 7 é bom mas não tem esse "before/after" instantâneo. Nosso Fig 2 (dose-response) é talvez mais impactante nesse sentido.

### Resumo de ações concretas

| Dimensão | Ação sugerida |
|---|---|
| Escrita | Título como pergunta seria mais memorável (mas não vamos mudar agora) |
| Claims | Estrutura "under X → bad; under Y → good" mais direta no abstract |
| Venda | "We correct the implicit assumption that FFT is the relevant paradigm" |
| Impacto | Nosso cenário (PEFT fine-tuning) é mais próximo da prática que pretraining |
| Precisão | Diferenciar claramente: eles fazem pretraining; nós fazemos fine-tuning |
| Memorabilidade | "Is collapse inevitable? Not if you control pressure." — frase candidata |

---

## Item 5: Zibakhsh et al. (2024) — ForTIFAI / TCE

*(próximo item — aguardando)*
