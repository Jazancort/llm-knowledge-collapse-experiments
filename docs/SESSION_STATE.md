# SESSION STATE — Post-Experiments Consolidated (2026-07-01 19:25)

## CRITICAL: Read this file after any context compaction.

---

## Project Overview

**Paper:** "Effective Training Pressure Gates Recursive Knowledge Degradation in LLMs: A Multi-Axis Dose-Response Study"
**Target:** Engineering Applications of Artificial Intelligence (Elsevier)
**Repo:** https://github.com/Jazancort/llm-knowledge-collapse
**Latest commit:** d9f4d8f (2026-07-01)
**Main file:** code/paper/cas-dc-template.tex (22 pages, compiles OK)
**Bib:** code/paper/cas-refs.bib (48 references cited)

---

## CURRENT STATUS: FINAL POLISH BEFORE SUBMISSION

All experiments COMPLETE. Paper integrates all results. Now in text-polish phase.

### Active TODO (lapidação):
1. Abstract: atualizar com rank×LR interaction, FFT cross-backbone, N=5, threshold refinado
2. Introduction contributions: atualizar lista para refletir novos resultados
3. Highlights: atualizar com interaction e cross-backbone FFT
4. Gerar heatmap Rank×LR (nova figura)
5. Revisar consistência: remover 'monotonic' onde não se aplica (FFT Gemma)
6. Compilar final e commit

---

## KEY DATA (all verified)

### Qwen 2.5 1.5B (K0=78-79)
- r=4: 96.2% Gen5 | r=16: 97.4% N=3 | r=64: 91.1% | r=128: 88.6% | r=256: 78.0% N=3

### Gemma 3 1B (K0=46-47)
- r=4: 91.3% mean N=5 (89.1-93.5%) | r=10: 78.3% | r=12: 73.9% | r=14: 69.6% | r=16: 69.1% mean N=5
- Threshold: erank 3-6 (10× lower than Qwen)
- Post-threshold plateau: r=16 ≈ r=256 ≈ 70%

### Rank × LR Matrix (Qwen, K0=78, Gen5)
| Rank \ LR | 5e-6 | 1e-5 | 2e-5 |
|---|---|---|---|
| r=16 | 97.4% | 97.4% | 92.3% |
| r=64 | 97.4% | 92.3% | 88.5% |
| r=256 | 89.7% | 84.6% | 82.1% |
**Axes interact. Not independent.**

### FFT Gemma 3 (K0=46, Gen5)
- LR=1e-6: 78.3% | LR=5e-6: 54.3% | LR=1e-5: 63.0%
- NOT monotonic (5e-6 worse than 1e-5) — single seed noise
- Magnitude axis generalizes cross-backbone

### FFT Qwen (K0=78, Gen10, N=3)
- QLoRA r=16: 97.4% | FFT LR=1e-6: 91.9% | Gap: +5.6pp

### Interventions (Qwen r=256, K0=78, Gen5)
- C3: 92.3% mean N=3 | C5: 92.7% mean N=3 masks
- Gemma 3 C5: NO EFFECT (same as baseline) — reported as negative result

### Negative Results
- Gemma 3 C5 (5% downsample at r=16): no improvement over baseline
- Exposure sensitivity currently demonstrated only for Qwen boundary case
- Documented in docs/NEGATIVE_RESULTS.md

---

## PAPER STRUCTURE (22 pages)

| Section | Content |
|---|---|
| Abstract | 4 findings (update to 5 with interaction) |
| Highlights | 5 items |
| 1. Introduction | 7 paragraphs + contributions |
| 2. Related Work | 5 subsections, 48 refs |
| 3. Methodology | 8 subsections + Fig methodology |
| 4. Results | 4.1 Rank sweep, 4.2 Cross-backbone, 4.3 FFT+interaction, 4.4 Distribution, 4.5 Intervention |
| 5. Discussion | 5.1 Framework+interaction, 5.2 Reconciliation, 5.3 Drift, 5.4 Practical |
| 6. Limitations | 10 items + normalization table + defensive paragraph |
| 7. Conclusion | 4 paragraphs |

## Tables (9)
1. Experimental axes
2. QLoRA Qwen rank sweep
3. Gemma 3 ranks (7 rows now)
4. FFT LR sweep Qwen
5. Interventions C1-C5
6. Evidence matrix (8 rows)
7. Generalization scope (7+7)
8. Rank × LR interaction (3×3)
9. Normalization attempts (6 rows inline)

## Figures (8)
1-2. Trajectories + dose-response (combined)
3. Cross-backbone
4. FFT vs QLoRA
5. Distributional signatures
6. Interventions
7. Methodology timeline
8. ETP framework (conceptual, figure*)

---

## WRITING RULES
1. No em-dashes. Commas or periods.
2. "suggests/indicates" not "confirms"
3. "empirical organizing framework" not "predictive/theoretical framework"
4. NOT "monotonic" for FFT Gemma (5e-6 > 1e-5 at Gen5)
5. "axes interact" not "pressure = rank × LR"
6. Contribution is qualitative (existence of transition), not quantitative (threshold values)
7. Exposure intervention specific to Qwen boundary, not universal

---

## Compilation
```bash
cd "G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\code\paper"
pdflatex → bibtex → pdflatex → pdflatex
```
