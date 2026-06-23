# Cronograma e Checklist Operacional

---

## Pré-requisitos

- [ ] Python 3.10+ instalado e no PATH
- [ ] CUDA 12.x instalado
- [ ] Ambiente virtual criado (conda ou venv)
- [ ] `pip install -r requirements.txt` executado sem erros
- [ ] GPU verificada: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- [ ] Espaço em disco: ~20GB para modelo + checkpoints

---

## M1A — Validação de Infraestrutura

**Tempo estimado:** 30-60 min  
**Objetivo:** confirmar que todas as medições funcionam

### Checklist

- [ ] Modelo carrega em 4-bit sem OOM
- [ ] Inferência produz texto coerente
- [ ] Hidden states extraídos (todas as camadas)
- [ ] Attention weights extraídos (todas as camadas)
- [ ] CKA self ≈ 1.0 (sanity check)
- [ ] ESI self ≈ 0.0 (sanity check)
- [ ] Accuracy > 0% nas 20 perguntas
- [ ] Log-prob e entropy calculados sem NaN
- [ ] Distinct-4 > 0
- [ ] metadata.json salvo

### Comando
```bash
python scripts/m1a_validate_infrastructure.py
```

### Critério de sucesso
Todos os sanity checks passam. Se algum falha: debugar antes de prosseguir.

---

## M1B — Ciclo Mínimo

**Tempo estimado:** 2-3h  
**Objetivo:** validar pipeline recursivo ponta a ponta (1 geração)

### Checklist

- [ ] QLoRA adapter criado corretamente
- [ ] Fine-tuning completa 2 epochs sem crash
- [ ] Train loss diminui ao longo do treinamento
- [ ] Merge do LoRA produz modelo funcional
- [ ] Modelo merged gera texto (não degenera)
- [ ] Geração de dados sintéticos funciona (Training Seed → respostas)
- [ ] Todas as métricas recalculadas no modelo pós-merge
- [ ] CKA(Gen0, Gen1) < 1.0 (alguma mudança ocorreu)
- [ ] ESI(Gen0→Gen1) > 0 (alguma instabilidade detectada)
- [ ] Accuracy registrada para Gen 0 e Gen 1
- [ ] Checkpoints salvos: model_gen0/, model_gen1/
- [ ] metadata.json com hashes de dataset e parâmetros

### Comando
```bash
python scripts/m1b_single_cycle.py
```

### Critério de sucesso
Pipeline executa sem crash. Métricas são numéricas e não-triviais (não todas zero ou NaN).

---

## M2 — Calibração (3 gerações, 1 seed, G1)

**Tempo estimado:** 6-8h  
**Objetivo:** ver se existe sinal + definir threshold GFW

### Checklist

- [ ] 3 gerações completas (Gen 0 → Gen 1 → Gen 2 → Gen 3)
- [ ] Accuracy registrada em cada geração (Evaluation Set)
- [ ] CKA registrado em cada geração (Probe Set)
- [ ] ESI registrado entre cada par de gerações
- [ ] Confidence e entropy registrados
- [ ] Distinct-4 registrado
- [ ] Gradient norm e parameter delta registrados
- [ ] Threshold GFW calculado: mean(ESI_0→1, ESI_1→2) + 2σ
- [ ] Threshold documentado e CONGELADO

### Perguntas a responder após M2

1. A accuracy está caindo? Se não: provavelmente precisa de mais gerações ou configuração mais agressiva.
2. O ESI está subindo? Se não: pode não haver sinal nesta escala.
3. O CKA está caindo? Se sim em quais camadas?
4. A fluência está se mantendo? (primeiros indícios de Stage B)
5. Quanto tempo leva cada geração? (projeção para M3)

### Decisão pós-M2

- Se existe sinal (ESI subindo, accuracy caindo): prosseguir para M3
- Se não existe sinal algum: revisar decisões (modelo? dataset? LR?)
- Se pipeline é muito lento: otimizar antes de M3

---

## M3 — Experimento Completo

**Tempo estimado:** 3-4 dias (paralelizável no lab)  
**Objetivo:** resultados publicáveis

### Configuração

```
10 gerações × 3 grupos × 3 seeds = 90 fine-tunings + merges
```

### Checklist

- [ ] Seed 42 completa (G1, G2, G3 × 10 gerações)
- [ ] Seed 137 completa (G1, G2, G3 × 10 gerações)
- [ ] Seed 256 completa (G1, G2, G3 × 10 gerações)
- [ ] Todas as métricas salvas por geração/grupo/seed
- [ ] Tabela de Lead Time preenchida
- [ ] Correlações ESI→Accuracy(t+1) calculadas
- [ ] Correlações baselines→Accuracy(t+1) calculadas
- [ ] Auditoria humana: 50 amostras × gerações selecionadas
- [ ] Cohen's Kappa calculado
- [ ] Visualizações geradas (figuras 1-5)
- [ ] Análise estatística completa

---

## Pós-M3 — Análise e Escrita

### Checklist

- [ ] Tabela principal: Lead Time por seed (com média e IC)
- [ ] Tabela comparativa: correlação preditiva de cada métrica
- [ ] Figuras finais: accuracy, CKA, ESI por geração/grupo
- [ ] Scatter plot: ESI(t) vs Accuracy(t+1)
- [ ] Narrativa definida: qual cenário ocorreu?
- [ ] Draft do artigo iniciado
- [ ] Revisão interna dos resultados

---

## Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| OOM durante fine-tuning | Média | Bloqueia M1B | Reduzir batch size, gradient checkpointing |
| Modelo não colapsa em 10 gerações | Baixa | Requer redesenho | Aumentar epochs ou LR por geração |
| G1 ≈ G2 (forgetting domina) | Média | Hipótese comprometida | Aumentar dataset ou reduzir LR |
| ESI muito ruidoso | Média | Resultado inconclusivo | Aumentar Probe Set (200→500) |
| GPU muito lenta para M3 | Alta | Prazo estourado | Usar computador do lab |
| Modelo piloto vs principal se comportam diferente | Média | M1 não prediz M3 | Aceitar — piloto é só validação de pipeline |

---

## Notas sobre Reprodutibilidade

Toda execução deve salvar:
```json
{
  "seed": 42,
  "model": "google/gemma-3-4b-it",
  "generation": 3,
  "group": "replacement",
  "temperature": 0.7,
  "learning_rate": 1e-5,
  "epochs": 2,
  "dataset_hash": "a1b2c3d4e5f6g7h8",
  "timestamp": "2026-06-25T14:30:00",
  "torch_version": "2.4.1",
  "transformers_version": "4.46.3",
  "cuda_version": "12.4",
  "gpu": "RTX 3070"
}
```

Isso garante que qualquer resultado pode ser rastreado até sua configuração exata.
