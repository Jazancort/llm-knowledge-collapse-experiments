# SESSION STATE — Full Consolidated (2026-06-29 19:13)

## CRITICAL: Read this file after any context compaction.
## Contains everything needed to continue writing the paper.

---

## Project Overview

**Paper:** "Effective Training Pressure Gates Recursive Knowledge Degradation in LLMs: A Multi-Axis Dose-Response Study"
**Target:** Engineering Applications of Artificial Intelligence (Elsevier, IF 7.8, A1 CAPES)
**Researcher:** Julio Azancort (UFPA)
**Repo:** https://github.com/Jazancort/llm-knowledge-collapse
**Latest commit:** 3c9b72b (2026-06-29)
**Main file:** code/paper/cas-dc-template.tex (13 pages double-column, compiles OK)

---

## Paper Progress

### COMPLETE (validated by user + GPT review):
- Section 3: Methodology (3.0-3.8) — all 8 subsections
- Section 4: Results (4.1-4.5 + opening + closing paragraphs)
- All 6 figures generated (PDF + SVG)
- cas-refs.bib with 23 references

### TODO (recommended writing order):
1. **Discussion** (5.1-5.4) — next to write
2. **Limitations** (Section 6, ~1 page)
3. **Conclusion** (Section 7, ~0.5 page)
4. **Introduction** (Section 1, ~1.5 pages)
5. **Related Work** (Section 2, ~2 pages)
6. **Abstract** (last)

---

## Final Paper Structure (VALIDATED)

```
1. Introduction
2. Related Work
   2.1 Recursive model collapse
   2.2 Knowledge degradation
   2.3 Mitigation strategies
   2.4 Parameter-efficient fine-tuning and forgetting
   2.5 Positioning of this work
3. Methodology               ✅
4. Results                   ✅
5. Discussion
   5.1 Effective training pressure as an organizing framework
   5.2 Relation to recursive-training literature
   5.3 Distributional degradation as a diagnostic signal
   5.4 Practical implications for recursive training systems
6. Limitations               (SEPARATE section, not inside Discussion)
7. Conclusion
```

---

## Central Thesis

Recursive synthetic fine-tuning stability is governed by **effective training pressure** (adapter capacity × update magnitude × synthetic exposure). There is a **sharp backbone-dependent threshold**: below it, the system is homeostatic; above it, factual degradation emerges with characteristic distributional signatures. The boundary is controllable: ~5% reduction in synthetic exposure shifts r=256 from degradative to near-homeostatic (tested on Qwen only).

---

## Key Results (ALL VERIFIED FROM JSONs)

### QLoRA Rank Dose-Response (Qwen 2.5 1.5B-Instruct, K0=79)
| Rank | Gen10 Ret | Eff Rank | Regime | Seeds |
|---|---|---|---|---|
| r=4 | 96.2% (Gen5) | 3.34 | Homeostatic | 1 |
| r=16 | 97.4% (Gen10, N=3) | 11.08 | Homeostatic | 3 |
| r=32 | 94.9% (Gen5) | 17.85 | Homeostatic | 1 |
| r=64 | 91.1% | 29.52 | Homeostatic | 1 |
| r=128 | 88.6% | 50.16 | Bounded | 1 |
| r=256 | 78.0% ±2.6% (Gen10, N=3) | 87.57 | Degradative | 3 |

Note: r=16 seeds have K0=78 (different run). r=256 mean = (75.9+77.2+81.0)/3/79.

### Cross-Backbone (Gemma 3 1B, K0=47)
| Rank | Gen5 Ret | Regime | Seeds |
|---|---|---|---|
| r=4 | 92.9% mean | Homeostatic | 3 |
| r=16 | 68.8% mean | Degradative | 3 |
| r=256 | 70.2% | Degradative (plateau) | 1 |

Threshold: eff rank ~3-9 (10× lower than Qwen ~50-88).
Post-threshold plateau: r=16 ≈ r=256 (~70%).

### Gemma 4 E2B (N=1, robustness only)
- r=4: 96.1%, eff 2.13 | r=16: 97.4%, eff 5.64 | Both homeostatic

### FFT vs QLoRA (K0=78, Gen10, N=3)
| Config | Seed 15 | Seed 137 | Seed 256 | Mean |
|---|---|---|---|---|
| QLoRA r=16 | 76/78 | 76/78 | 76/78 | 97.4% |
| FFT LR=1e-6 | 72/78 | 71/78 | 72/78 | 91.9% |
| **Gap** | 5.1pp | 6.4pp | 5.1pp | **+5.6pp** |

- Gap is one-time adaptation cost (emerges Gen1, flat after)
- FFT fragile-fact floor: same 6 facts lost cross-seed (Jaccard=1.0)
- QLoRA: 2-3 facts lost, different ones (Jaccard=0.0 vs FFT)
- FFT dose-response: LR 1e-6→2e-5 yields 92.3%→84.6% (drift dominates)

### Distributional Signatures
- Three-regime taxonomy: homeostatic / distributionally degraded / factually degradative
- r=128 dissociation: retention 88.6% bounded BUT content efficiency 0.13, persistence 5.2%, MTLD 745
- Two phenotypes: r=128 "filler verbosity" vs r=256 "elaborative drift"
- Cross-lag: raw ρ=-0.965, partial ρ=-0.22, first-diff=-0.43 → co-symptom

### Interventions (Qwen r=256, K0=78, Gen5)
| Condition | Retention | Delta vs C1 | Tokens | N |
|---|---|---|---|---|
| C1 Normal | 65/78 (83.3%) | — | ~57k | 1 |
| C2 Short | 71/78 (91.0%) | +7.7pp | ~60k | 1 |
| C3 Filtered | 72.0/78 mean (92.3%) | +9.0pp | ~54k | 3 seeds |
| C4 Canonical | 69/78 (88.5%) | +5.1pp | ~52k | 1 |
| C5 Random | 72.3/78 mean (92.7%) | +9.4pp | ~54k | 3 masks |

- C3 per-seed: 73, 71, 72 /78
- C5 per-mask: 72, 73, 72 /78
- Jaccard C3 vs C5 removed sets: 0.015
- C2 uses MORE tokens but still improves → not token-budget confound

---

## Discussion Plan (validated by user+GPT)

### 5.1 Effective training pressure as organizing framework
- Three axes converge: rank (capacity), LR (magnitude), exposure (volume)
- All shift the same regime transition
- Framework is conceptual, not a fitted equation
- "Whether pressure is increased through capacity, magnitude, or exposure, the system responds qualitatively in the same way"
- Reserve the formal synthesis for HERE (removed from Results closing)

### 5.2 Relation to recursive-training literature
- Shumailov/Dohmatob: we locate regime boundaries where their predictions hold, not refute
- Keisha: same backbone (Gemma 3 1B) suggestive but different protocol
- Biderman: we map WHEN "forgets less" holds vs fails
- Gerstgrasser: orthogonal axis (data accumulation vs update pressure)
- Tone: "complements and refines", not "contradicts"

### 5.3 Distributional degradation as diagnostic signal
- r=128 dissociation deserves explicit discussion (most interesting result after dose-response)
- Output drift accompanies degradation but not demonstrated as causal
- Practical monitoring value: length, content efficiency, persistence
- "Factual accuracy alone does not fully capture the onset of degradation"

### 5.4 Practical implications for recursive training systems
- Rank as a control knob
- LR as a control knob
- Exposure as a control knob
- SDI/content-efficiency/persistence as monitoring signals
- Rule of thumb: if indicators degrade, reduce pressure
- Connects directly to EAAI engineering scope

---

## Limitations Plan (~1 page, Section 6)

1. Model scale: 1-2B only, behavior at 7B/70B unknown
2. Single dataset: TriviaQA factoid QA only
3. N=3 for headline claims, N=1 for ablations (module topology, Gemma 4)
4. Gen10 maximum horizon, long-term dynamics unknown
5. Cross-architecture threshold normalization unsolved
6. Perturbation matching approximate (||B@A|| vs weight drift, not identical)
7. Intervention C3/C5 tested only at r=256 boundary on Qwen
8. No weight-level mechanism analysis (output-level only)
9. Output metrics descriptive, not causal
10. Only 2 backbones with threshold located (Qwen, Gemma 3)

---

## Conclusion Plan (~0.5 page, Section 7)

1. Summary: recursive degradation is pressure-gated, sharp boundary
2. Main contributions (dose-response, cross-backbone, FFT comparison, diagnostic signatures, intervention)
3. Implication: boundary is sharp and controllable (not inevitable)
4. Future work (1 paragraph): scale, datasets, causal decomposition, normalization, combine axes

---

## Writing Rules (CRITICAL)

1. **NEVER use em-dashes (---)** as punctuation. Always commas or periods.
2. Write section here first → user validates with GPT → then insert in .tex
3. No overclaims: every assertion checked against outputs/ JSONs
4. "suggests/indicates" not "confirms"
5. "approximately perturbation-matched" not "exactly matched"
6. "in the Qwen r=256 boundary case" not "any 5% universally works"
7. "identifies regime boundaries" not "contradicts prior work"
8. "diagnostic signature" not "early warning" (temporal precedence not proven)
9. "We refer to this as..." (our terms) not "This represents..." (established taxonomy)
10. Cross-lag: "co-symptom" not "causal antecedent"

---

## EAAI-Specific Requirements

- **Highlights** (3-5 bullets, mandatory): already planned
- **Keywords:** 4-6
- **Abstract:** 150-250 words, no equations, no references
- **Scope:** novel AI aspect + real engineering application + public dataset
- **References:** 30-60 typical, numbered with DOIs
- **CRediT:** required
- **Data availability:** GitHub repo link
- **Template:** cas-dc.cls (double-column), system TeX Live 2025

---

## Figures (all in paper/figs/)

| Fig | File | Content | Referenced in |
|---|---|---|---|
| 1 | fig1_trajectories.pdf | Longitudinal r=16/128/256 Gen0-10 | §4.1 |
| 2 | fig2_dose_response.pdf | Retention + eff rank vs nominal rank | §4.1 |
| 3 | fig3_cross_backbone.pdf | Shapes=backbone, colors=regime | §4.2 |
| 4 | fig4_fft_vs_qlora.pdf | FFT vs QLoRA Gen10, 3 seeds | §4.3 |
| 5 | fig5_distributional.pdf | Length/persistence/efficiency (3 panels) | §4.4 |
| 6 | fig6_interventions.pdf | C1-C5 retention + delta (2 panels) | §4.5 |

---

## References (23 entries in cas-refs.bib)

**Core collapse:** shumailov2024, dohmatob2025, seddik2024, xu2025, alemohammad2024
**Knowledge:** keisha2025
**Mitigation:** gerstgrasser2024, zibakhsh2024, yi2025, adapala2025
**PEFT:** hu2022, dettmers2023, biderman2024, aghajanyan2021
**Effective rank:** roy2007
**Metrics:** li2016 (Distinct-n), mccarthy2010 (MTLD), holtzman2020
**Data:** joshi2017 (TriviaQA), villalobos2024
**Models:** qwen2024, gemma2025
**Other:** dey2024

---

## Compilation

```bash
cd "G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\code\paper"
pdflatex -interaction=nonstopmode cas-dc-template.tex
```

Uses system TeX Live 2025 (cls from c:\texlive\2025\texmf-dist\tex\latex\els-cas-templates\).

---

## Technical Notes

- K0=79 for g1_rank* runs (Qwen main sweep)
- K0=78 for fft_drift_gen10 and all intervention runs (different random eval split)
- K0=47 for Gemma 3 runs
- Full-linear r=4 mean eff rank = 3.43 (196 matrices)
- Qwen architecture: 28L, 1536d, 12h, 2kv
- Gemma 3 architecture: 26L, 1152d, 4h, 1kv
- Loss: DataCollatorForLanguageModeling mlm=False (full prompt+response)
- Training: 2 epochs, LR 1e-5, batch 4, gradient_accumulation 2
- Generation: T=0.7, max_new_tokens=50
- Seeds: 15, 137, 256 (for multi-seed runs)

---

## Other Docs in code/docs/

| File | Content | Relevance |
|---|---|---|
| PROJECT_STATUS_v2.md | ALL experimental results + claims + positioning | Source of truth for data |
| PAPER_OUTLINE.md | Figures/tables plan + wording decisions | Writing guide |
| PAPER_STRUCTURE.md | Detailed subsection breakdown | Outline reference |
| REFERENCES_SEARCH.md | 23 refs with abstracts and DOIs | For Related Work |
| TEMPLATE_RULES.md | Elsevier CAS-DC formatting rules | Submission checklist |
| FUTURE_WORK.md | 18 future directions organized by tier | For Conclusion paragraph |
| DECISIONS.md | PARTIALLY OBSOLETE design decisions | Historical only |
| PROTOCOL.md | OBSOLETE (pre-pivot protocol) | Historical only |
| THEORY.md | OBSOLETE (ESI hypothesis, abandoned) | Historical only |
| SCENARIOS.md | OBSOLETE (pre-pivot scenarios) | Historical only |
| METRICS.md | OBSOLETE (pre-pivot metric definitions) | Historical only |
| GRILL_SESSION*.md | Session logs from initial design | Historical only |
| CHECKLIST.md | Old experimental checklist | Historical only |
| PHASE2_PLAN.md | Old Phase 2 plan | Historical only |
