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
