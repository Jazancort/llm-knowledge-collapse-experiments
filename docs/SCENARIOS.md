> **⚠️ OBSOLETE** — These scenarios describe outcomes for the ESI/Lead-Time hypothesis.
> The project found capacity-gated regime transitions instead.
> See `PROJECT_STATUS.md` for actual results.

# Cenários de Resultado e Interpretação

Este documento define a priori como cada cenário de resultado será interpretado. Escrito ANTES de rodar o experimento para evitar p-hacking narrativo.

---

## Cenário 1 — Resultado ideal

**Observação:**
- Lead Time ≥ 2 gerações (ESI sobe na Gen 3, accuracy cai na Gen 5+)
- Consistente nas 3 seeds
- ESI tem correlação com Accuracy(t+1) superior a todas as baselines
- Stage B observável (fluência estável por ≥1 geração após queda de accuracy)

**Interpretação:** H1 confirmada, H2 confirmada, H3 confirmada.

**Artigo:** contribuição forte. Submissão direta para venue de IA confiável / XAI / AI Safety.

**Framing:** "Explanation stability analysis provides a robust early warning signal for knowledge collapse, detecting representational drift 2+ generations before observable factual degradation."

---

## Cenário 2 — Resultado parcial (Lead Time ≈ 1)

**Observação:**
- ESI sobe ligeiramente antes da accuracy cair, mas por apenas ~1 geração
- Consistente nas seeds, mas variância alta
- ESI ligeiramente melhor que baselines, mas não dramaticamente
- Stage B fraco ou ausente

**Interpretação:** H1 parcialmente confirmada. H2 marginal. H3 rejeitada ou inconclusiva.

**Artigo:** publicável com framing adequado.

**Framing:** "Representational instability provides marginal but consistent early indication of impending factual degradation under recursive training."

**Venues:** workshops de AI Safety, conferências menores, journal de escopo mais amplo.

---

## Cenário 3 — Resultado negativo (Lead Time ≈ 0)

**Observação:**
- CKA e accuracy caem simultaneamente
- ESI não tem poder preditivo superior às métricas tradicionais
- Tudo degrada junto, sem precedência temporal

**Interpretação:** H1 rejeitada. ESI não é early warning.

**O que fazer:**
1. Verificar se o efeito aparece em camadas específicas (CKA por camada individual)
2. Verificar se G1 ≫ G2 em degradação (contribuição sobre collapse vs forgetting persiste)
3. Pivotar para: "quais métricas correlacionam melhor com degradação factual sob recursive training?"
4. Resultado negativo publicável se consistente: "Attention-based explanation stability does not provide advance warning of knowledge collapse in models of this scale."

**Artigo:** mais fraco, mas ainda possui contribuição empírica (reprodução de collapse + comparação de detectabilidade).

---

## Cenário 4a — Inesperado: CKA estável mas accuracy cai

**Observação:**
- Hidden states (representações) se mantêm estáveis entre gerações
- Mas o modelo passa a errar mais

**Interpretação:** O colapso está ocorrendo na camada de decodificação/output, não na representação. As features "estão lá" mas o modelo não as usa corretamente.

**O que fazer:**
- Investigar attention drift (se atenção muda mas CKA não)
- Olhar logit lens / output layer weights
- Nova hipótese: "collapse is a decoding-layer phenomenon, not a representational one"

**Artigo:** resultado surpreendente e potencialmente muito interessante. Requer investigação adicional antes de publicar claim forte.

---

## Cenário 4b — Inesperado: CKA cai muito cedo (Gen 1) mas accuracy cai tarde (Gen 7+)

**Observação:**
- Representações divergem drasticamente logo na primeira geração
- Mas o modelo continua respondendo corretamente por várias gerações depois

**Interpretação:** Lead Time muito grande. O modelo tem resiliência factual mesmo com representações alteradas. Eventualmente a degradação representacional se manifesta em outputs.

**Artigo:** provavelmente o resultado MAIS interessante possível. Demonstra claramente que monitoramento representacional pode fornecer antecedência significativa.

**Cuidado:** verificar se não é artefato do processo de merge (primeira geração sempre produz grande mudança nos pesos).

---

## Cenário 5 — G1 ≈ G2 (forgetting domina)

**Observação:**
- Grupo Replacement degrada na mesma taxa que grupo Real-only
- Não há diferença significativa entre treinar com dados sintéticos vs reais

**Interpretação:** O efeito observado NÃO é knowledge collapse. É catastrophic forgetting por fine-tuning iterativo.

**O que fazer:**
- Aumentar dataset (talvez 5k era pouco)
- Testar com mais epochs (talvez 2 epochs não é suficiente para imprimir os dados)
- Reduzir LR (talvez 1e-5 é agressivo demais para esse modelo)
- Se persiste após ajustes: resultado sobre forgetting, não sobre collapse

**Artigo:** pivot necessário. Publicável como contribuição sobre dinâmicas de forgetting em fine-tuning iterativo, mas não responde a pergunta original.

---

## Cenário 6 — Modelo não colapsa em 10 gerações

**Observação:**
- Accuracy se mantém relativamente estável (queda < 10pp) mesmo após 10 gerações de replacement
- ESI pode ou não subir

**Interpretação:** O modelo escolhido é resiliente demais para o dataset e configuração usados. Ou: 10 gerações não são suficientes para este modelo/escala.

**O que fazer:**
- Verificar se respostas estão ficando mais homogêneas (D4 caindo) mesmo sem queda de accuracy
- Aumentar agressividade: mais epochs por geração, LR mais alto
- Reduzir dataset (menos dados por geração amplifica o efeito)
- Trocar para modelo menor se necessário

---

## Tabela de Decisões por Cenário

| Cenário | Lead Time | H1 | Artigo | Ação |
|---|---|---|---|---|
| 1 (ideal) | ≥ 2 | ✓ | Forte | Submeter |
| 2 (parcial) | ≈ 1 | ≈ | Moderado | Submeter com framing conservador |
| 3 (negativo) | ≈ 0 | ✗ | Fraco | Pivotar ou publicar negativo |
| 4a (CKA ok, acc cai) | N/A | ✗ | Surpreendente | Investigar camada de output |
| 4b (CKA cai cedo) | ≫ 2 | ✓✓ | Muito forte | Submeter com claim forte |
| 5 (forgetting) | N/A | ✗ | Pivot | Redesenhar ou publicar forgetting |
| 6 (sem collapse) | N/A | N/A | Inconclusivo | Ajustar agressividade |

---

## Gate Obrigatório: Adapter Health (pré-condição para interpretar M3)

**ANTES de interpretar qualquer resultado de CKA/ESI/accuracy, verificar:**

| Métrica | Condição ideal (Cenário A) | Sinal de alerta (Cenário B) |
|---|---|---|
| Effective Rank | Variação < 20% do valor em Gen 0 | Queda > 50% |
| Spectral Norm | Sem crescimento explosivo (< 3× Gen 0) | Explosão ou colapso |

**Se Cenário A (adapter saudável):** Interpretar CKA/ESI/accuracy normalmente. H1 testável.

**Se Cenário B (rank collapse):** NÃO parar. Reformular:
- Medir se G1 e G2 diferem no effective rank (se ambos caem = artefato de FT iterativo, não de recursão)
- Se apenas G1 cai: o rank collapse é parte da cadeia de degradação (dados → adapter → representação → accuracy)
- Nova pergunta: "Qual é a relação entre entropia dos dados, capacidade do adapter e degradação factual?"
- Experimento de mitigação posterior: forçar entropia alta (T>1.0, penalties) e ver se H1 se sustenta

**Condição de falha REAL (artigo não publicável como Knowledge Collapse):**
```
G1 ≈ G2 em TODAS as métricas (effective rank, CKA, accuracy)
```
Nesse caso: tudo é catastrophic forgetting + adapter degeneration. Recursive collapse não detectado.
