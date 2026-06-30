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


## Item 5: Zibakhsh et al. (2024) — ForTIFAI / TCE

*"ForTIFAI: Fending Off Recursive Training Induced Failure for AI Model Collapse"*
*Preprint, UCSD + Stanford*

### Escrita

**O que fazem bem:**
- O acrônimo "ForTIFAI" é catchy e memorável (Fortify + AI). Marketing acadêmico eficiente.
- A insight central é comunicada em 1 frase: "auto-regressive models tend to generate text to which they assign high confidence." Simples, verificável, intuitivo.
- A contribuição é ultra-prática: "a loss function." É algo que qualquer engenheiro pode implementar amanhã.
- "2.3× more synthetic data before collapse" — número claro, benchmark-able.
- Figure 1 mostra distribuição de confiança (real vs synthetic) — prova visual instantânea da insight.

**O que fazem mal:**
- O paper é preprint (sem peer review ainda).
- A escrita é boa mas não excepcional (padrão ICLR workshop).
- A teoria é limitada (observação empírica + loss function, sem prova de convergência).

**Lição para nosso paper:**
- A força deles é a **simplicidade actionable**: 1 insight → 1 loss function → 1 número (2.3×).
- Nosso equivalente actionable é: "1 knob (rank) → 1 threshold → 1 number (5% to flip)."
- Mas nós temos MAIS knobs e MAIS generalizáveis. A vantagem é que mostramos o landscape inteiro, não apenas 1 hack.

### Claims

**O que fazem:**
- "Model's confidence is a strong signal for identifying its own generated text." — verificável e surpreendente.
- "TCE tolerates 2.3× more synthetic data without degradation." — benchmark claro.
- "Model-agnostic, computationally efficient, easy to implement." — vende para engenheiros.
- Também contribuem: "comprehensive evaluation framework for model collapse" (benchmark open-source).

**O que fazem mal:**
- Não provam por que TCE funciona teoricamente. É puramente empírica.
- Não testam sob PEFT. Só FFT.
- O "2.3×" é sobre mixed-data proportion tolerance, não sobre recursion depth.

**Lição para nosso paper:**
- O claim "model-agnostic, computationally efficient, easy to implement" é exatamente o tipo de frase que EAAI adora.
- Nós podemos dizer algo análogo: "Adapter rank selection requires no additional infrastructure, no external data, and no changes to the optimization procedure." Isso posiciona nosso achado como igualmente prático.
- A diferença: TCE é uma **intervenção** (muda como treinar). Nosso achado é uma **observação + framework** (mostra quando treinar é seguro). São complementares.

### Venda / Impacto

**O que fazem:**
- Vendem como "practical tool for deployment." Não como ciência básica.
- O open-source benchmark é uma contribuição tangível (reprodutibilidade).
- A afiliação UCSD + Stanford + Azalia Mirhoseini (nome de peso em ML systems) ajuda.
- Posicionamento: "while causes are understood, mitigation is scarce." Cria urgência para a solução.

**Lição para nosso paper:**
- Nós também temos código open-source (GitHub). Devemos mencionar isso mais prominentemente.
- A frase "mitigation strategies remain scarce" poderia ser adaptada para nós: "governance frameworks for recursive PEFT remain absent." 
- Para EAAI, o fato de termos resultados **sem modificar a loss** é uma vantagem: nosso approach é zero-cost (só escolher rank correto).

### Precisão

**O que fazem:**
- LLaMA-3.2-1B como modelo principal.
- WikiText como dataset. Mixed-data settings (variando proporção de synthetic).
- Benchmark com métricas de retained + acquired knowledge.
- Comparação com CE padrão ao longo de gerações.

**O que fazem mal:**
- Não fazem recursion pura (replace-only). Fazem mixed-data com proporções crescentes.
- Não isolam rank/capacity como variável.
- O cenário é "quanto synthetic tolero?" não "quando o recursive loop degrada?"

**Lição para nosso paper:**
- Cenários diferentes: eles medem tolerância a synthetic proportion; nós medimos dinâmica recursiva completa.
- Nós somos mais agressivos no protocolo (100% replace, zero real data). Isso é mais exigente e mais informativo.
- O posicionamento "eixo ortogonal" no Related Work está correto: TCE opera na loss; nós operamos no update mechanism.

### Memorabilidade

**O que faz esse paper ser lembrado:**
1. "ForTIFAI" — acrônimo catchy.
2. "2.3× more synthetic data" — número claro e impressionante.
3. "Ignore what the model already knows too well" — insight intuitiva.
4. Figure 1: histograma de confiança (visual proof of concept).
5. "Minimal extension of CE" — baixa barreira de adoção.

**Lição para nosso paper:**
- Nosso paper não tem acrônimo nem nome memorável para o método/framework. "Effective training pressure" é funcional mas longo. "ETP" como abreviação? Provavelmente não vale criar.
- O visual proof of concept equivalente é nosso **Fig 2 (dose-response curve)**: queda monotônica clara = proof visual.
- A "baixa barreira de adoção" para nós é: "just pick a lower rank." Zero cost, zero infrastructure change.

### Resumo de ações concretas

| Dimensão | Ação sugerida |
|---|---|
| Escrita | Adicionar frase tipo "requires no changes to optimization or data pipeline" |
| Claims | Posicionar rank selection como zero-cost governance (sem modificar loss, dados, ou pipeline) |
| Venda | Mencionar open-source code + reprodutibility como contribuição tangível |
| Impacto | Nosso approach é complementar ao deles: rank governance + TCE poderiam ser combinados |
| Precisão | Nosso protocolo (100% replace) é mais exigente que o mixed-data deles |
| Memorabilidade | Fig 2 (dose-response) é nosso visual proof of concept — dar destaque |

---

## Item 6: Yi et al. (2025) — Verification

*(próximo item — aguardando)*


## Item 6: Yi et al. (2025) — Verification

*"Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence"*
*arXiv:2510.16657, University of Chicago*

### Escrita

**O que fazem bem:**
- O título promete escape ("Escaping") e entrega teoria + empiria. Verbo ativo no título.
- A dualidade "near-term improvements / long-term convergence" no subtítulo comunica que o paper tem nuance (não é simplista).
- A Figure 1 (VAE MNIST: com verifier vs sem verifier) é uma prova visual instantânea. Leitor entende o paper em 3 segundos olhando a figura.
- O conceito de "verifier's knowledge center" é elegante: o modelo converge para o que o verifier sabe, não para a verdade.
- A insight "early gains will plateau and may reverse" é honesta e diferenciadora. Não vendem a solução como perfeita.

**O que fazem mal:**
- Abstract um pouco longo e repetitivo.
- O cenário principal é linear regression + VAE/MNIST. Não testam LLMs full-scale.

**Lição para nosso paper:**
- A honestidade sobre limitações da própria solução ("gains plateau, may reverse") é uma estratégia brilhante: desativa o reviewer que ia atacar overclaim. Nós fazemos algo similar com "tested only at boundary, not claimed universal."
- A Figure 1 estilo "antes/depois com e sem intervenção" é poderosíssima. Nosso equivalente seria: trajectories r=16 (stable) vs r=256 (degrades) — que é nosso Fig 1. Está bom.

### Claims

**O que fazem:**
- "By injecting information through an external verifier, synthetic retraining will not cause collapse." — forte, com prova.
- "Unless verifier is perfectly reliable, early gains will plateau." — nuance teórica.
- Teorema 3.1 (short-term improvement) + Teorema 4.1 (long-term convergence to θc).
- Contraction mapping framework: verifier transforma iteração em contração → convergência garantida.

**O que fazem mal:**
- O claim "will not cause collapse" depende de ter um verifier melhor que o modelo. Isso é uma assunção forte não sempre satisfeita.
- Não testam em cenário de recursive fine-tuning com LLMs.

**Lição para nosso paper:**
- Eles provam que external information injection prevents collapse. Nós mostramos que internal constraint (low rank) prevents collapse. São mecanismos diferentes com o mesmo efeito.
- A formulação "contraction mapping" é elegante. Nós não temos teoria equivalente, mas temos dose-response empírico que é mais diretamente aplicável para engenheiros.
- O "verifier paradox" (convergence to θc, not θ*) é análogo ao nosso achado de que low-rank QLoRA preserva knowledge mas limits learning capacity. Ambos são trade-offs.

### Venda / Impacto

**O que fazem:**
- Vendem como "the first to show verifier fundamentally alters long-term dynamics."
- Posicionamento teórico forte (Schmidt Sciences funding, UChicago statistics).
- O paper tem um resultado positivo E um resultado de cautela. Ambos são contribuições.

**Lição para nosso paper:**
- Nós também temos resultado positivo (stability under low pressure) E resultado de cautela (r=128 hidden degradation). Esse padrão "good news + warning" é poderoso.
- A frase "the first to formally show that..." é uma reivindicação de prioridade. Nós evitamos "the first" (corretamente), mas podemos dizer "to our knowledge, the first systematic dose-response characterization" — o que já fazemos.

### Precisão

**O que fazem:**
- Teoria: linear regression com provas formais (contraction, supermartingale).
- Empiria: linear regression simulation + VAE on MNIST.
- Não testam: LLMs, recursive fine-tuning, factual QA, PEFT.

**O que fazem mal:**
- O gap entre a teoria (linear models) e prática (LLMs) não é bridged.
- O "verifier" é assumido como disponível e melhor que o modelo. Em prática de recursive fine-tuning, isso nem sempre existe.

**Lição para nosso paper:**
- Nosso approach não requer verifier externo. O rank é um constraint intrínseco. Isso é uma vantagem prática significativa que devemos articular: "no external verifier or oracle needed."
- A comparação: eles precisam de algo externo (verifier); Gerstgrasser precisa de algo externo (real data); TCE modifica a loss. Nós não precisamos de nada externo — só configurar corretamente.

### Memorabilidade

**O que faz esse paper ser lembrado:**
1. "Verifier's knowledge center" — conceito elegante.
2. "Near-term improvement, long-term plateau" — dualidade honesta.
3. Figure 1: MNIST com/sem verifier (visual proof instantâneo).
4. "Contraction mapping" — framework teórico limpo.
5. "Escaping" no título — promessa de solução.

**Lição para nosso paper:**
- Nosso conceito equivalente a "knowledge center" é "pressure threshold" — o ponto para o qual o sistema converge (homeostasis) ou do qual diverge (degradation).
- A dualidade "near-term/long-term" deles é análoga à nossa "bounded/degradative." Ambos capturam que há regimes diferentes dependendo das condições.
- A grande vantagem nossa: não precisamos de nada externo. A estabilidade é uma propriedade intrínseca do sistema quando a pressão é baixa.

### Resumo de ações concretas

| Dimensão | Ação sugerida |
|---|---|
| Escrita | Enfatizar que nosso approach é "intrinsic" (não precisa de verifier, real data, ou loss change) |
| Claims | "Internal constraint (low rank) achieves similar stability to external verification" |
| Venda | Contraste: verification/accumulation/TCE = external; rank governance = internal |
| Impacto | Posicionar como a alternativa zero-cost: nenhuma infraestrutura adicional |
| Precisão | Testamos em LLMs reais com factual QA; eles testam linear + VAE/MNIST |
| Memorabilidade | "Pressure threshold" é nosso equivalente a "knowledge center" |

---

## Item 7: Xu et al. (2025) — Probabilistic Perspective

*(próximo item — aguardando)*


## Item 7: Xu et al. (2025) — Probabilistic Perspective

*"A Probabilistic Perspective on Model Collapse"*
*arXiv, 2025*

### Escrita

**O que fazem bem:**
- O título é preciso e posicional: "A Probabilistic Perspective" sinaliza que estão oferecendo um novo framework teórico, não apenas mais um resultado empírico.
- A formulação de recursive training como random walk é elegante e intuitiva. Qualquer pessoa com estatística básica entende a metáfora.
- Os resultados são formulados como questões (Q1, Q2) com respostas formais. Estrutura pergunta → teorema.
- O resultado P(T) < 1/2 é comunicado de forma limpa: "the probability of improvement is strictly less than one-half." Memorável.

**O que fazem mal:**
- O paper é denso. Muitos teoremas, corolários, lemas. Audiência técnica restrita.
- Pouca empiria em modelos reais (foco em Gaussian estimation).
- A apresentação poderia ser mais acessível para engenheiros.

**Lição para nosso paper:**
- O random walk framework é uma metáfora poderosa. Nosso "effective training pressure" opera na mesma intuição: cada geração é um passo, e o tamanho do passo depende da pressão.
- Mas eles formalizam; nós operacionalizamos. São contribuições complementares.
- A clareza de "P(T) < 1/2" — uma estatística única que resume o paper — é algo que devemos emular. Nosso equivalente: "threshold differs by 10× across backbones" ou "5% shifts the regime." Números únicos que resumem.

### Claims

**O que fazem:**
- "Superlinear sample growth prevents collapse." — Generaliza Shumailov.
- "P(improvement) < 1/2 regardless of sample schedule." — Resultado negativo forte.
- "Biased estimation accelerates collapse." — Insight mecanística.
- Contribuição taxonômica: small bias vs large bias regimes com thresholds diferentes.

**O que fazem mal:**
- Tudo em Gaussian/parametric models. Não provam que vale para LLMs.
- O gap teoria → prática é grande.

**Lição para nosso paper:**
- O resultado "P(T) < 1/2" é consistente com nosso achado: no regime degradativo, retenção cai monotonicamente. Cada geração tem mais chance de perder do que ganhar fatos.
- No regime homeostático, essa probabilidade parece estar acima de 1/2 (retenção estável ou melhora após Gen1). Isso é um insight que podemos articular na Discussion.
- Podemos dizer: "The random-walk framework of Xu et al. predicts that improvement probability is bounded below 1/2. Our homeostatic regime is consistent with configurations where step size is small enough to keep the walk bounded."

### Venda / Impacto

**O que fazem:**
- Vendem como a "base teórica rigorosa" para o que outros observaram empiricamente.
- Conectam com estimation theory clássica (bias-variance tradeoff).
- Positioning: "We generalize Shumailov et al. as a special case."

**Lição para nosso paper:**
- Eles generalizam a teoria. Nós operacionalizamos a prática. São lados complementares.
- Para EAAI, nosso framing prático é mais valioso. Para ICLR/NeurIPS, o deles seria.
- Podemos usar o resultado P(T)<1/2 para contextualizar por que o regime degradativo degrada monotonicamente: "consistent with the theoretical prediction that recursive improvement probability is bounded below 1/2."

### Precisão

**O que fazem:**
- Provas formais com concentration bounds, supermartingale inequalities.
- Framework geral: qualquer classe paramétrica com estimation procedure M.
- Conditions explícitas: Assumptions 1, 2, 3 (unbiased, convergence rate, bias bounds).
- Corollaries com closed-form para Gaussian case.

**O que fazem mal:**
- Não experimentam com neural networks ou LLMs.
- As assumptions (especially unbiasedness) podem não valer para fine-tuning.

**Lição para nosso paper:**
- As assumptions deles (unbiased estimation, parametric model) não valem diretamente para LoRA fine-tuning. Mas a intuição (random walk, step size = f(sample size)) mapa para nosso framework: step size ≈ f(rank, LR).
- Não precisamos provar que nosso cenário satisfaz as assumptions deles. Basta dizer que a intuição é consistente.

### Memorabilidade

**O que faz esse paper ser lembrado:**
1. "P(T) < 1/2" — resultado único, limpo, citável.
2. "Random walk of model parameters" — metáfora intuitiva.
3. "Superlinear sample growth" como threshold — prescription clear.
4. Generaliza Shumailov formalmente.
5. Bias-variance tradeoff aplicado a recursive training.

**Lição para nosso paper:**
- Nosso equivalente a "P(T) < 1/2" como single-number summary:
  - "Retention declines ~2pp/generation in degradative regime" (taxa constante)
  - "Threshold effective rank: ~50 on Qwen, ~5 on Gemma" (números precisos)
  - "5% exposure reduction: +9pp recovery" (intervention effect)
- Devemos garantir que pelo menos 1 desses seja o "número de referência" do paper. O que melhor resume tudo provavelmente é o contraste de threshold: "10× difference across backbones."

### Resumo de ações concretas

| Dimensão | Ação sugerida |
|---|---|
| Escrita | Usar a metáfora random walk na Discussion como conexão teórica |
| Claims | "Consistent with P(T)<1/2: degradative regime shows monotonic decline" |
| Venda | Complementar: eles formalizam o why; nós mapamos o where/when |
| Impacto | Para EAAI, operacionalização > teoria. Estamos no venue certo |
| Precisão | Não precisamos satisfazer assumptions deles; basta conexão intuitiva |
| Memorabilidade | Escolher 1 número como "the number" do paper (10× threshold diff?) |

---

## Síntese Final

*(será preenchida após todos os items)*
