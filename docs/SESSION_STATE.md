# SESSION STATE — Consolidated (2026-06-29 19:08)

## Project: Recursive Synthetic Fine-Tuning — Pressure-Gated Degradation
## Researcher: Julio Azancort (UFPA)
## Status: WRITING PHASE. Methodology + Results complete. Discussion/Intro/Conclusion pending.
## Target: Engineering Applications of Artificial Intelligence (Elsevier, IF 7.8, A1)
## Latest commit: 3c9b72b (2026-06-29)
## Repo: https://github.com/Jazancort/llm-knowledge-collapse

---

## Paper Progress

### COMPLETE:
- Section 3: Methodology (3.0-3.8) — 8 subsections, all validated by GPT
- Section 4: Results (4.1-4.5 + opening + closing) — 5 subsections, all validated
- All 6 figures generated (PDF + SVG, SciencePlots style)
- .bib with 23 references
- Template compiles: 13 pages double-column

### TODO (in recommended writing order):
1. Discussion 5.1-5.4
2. Limitations (Section 6)
3. Conclusion (Section 7)
4. Introduction (Section 1)
5. Related Work (Section 2)
6. Abstract (last)

---

## Final Paper Structure

```
1. Introduction              ← TODO
2. Related Work              ← TODO
   2.1 Recursive model collapse
   2.2 Knowledge degradation
   2.3 Mitigation strategies
   2.4 Parameter-efficient fine-tuning and forgetting
   2.5 Positioning of this work
3. Methodology               ✅ DONE
   3.0 Opening (pressure framework)
   3.1 Recursive replace-only protocol
   3.2 K0 factual retention metric
   3.3 Models and hardware
   3.4 Experimental axes (with table)
   3.5 Effective training pressure (equations: erank, drift)
   3.6 Output-distribution diagnostics (equations: efficiency, persistence, SDI)
   3.7 Synthetic-exposure interventions (C1-C5)
   3.8 Seeds, statistics, and reporting
   3.closing Bridge paragraph
4. Results                   ✅ DONE
   4.opening Paragraph
   4.1 Update capacity: QLoRA rank dose-response (Table + Fig 1)
   4.2 Cross-backbone thresholds (Table + Fig 3)
   4.3 Perturbation magnitude: FFT (Table + Fig 4)
   4.4 Distributional signatures (Fig 5, three-regime taxonomy)
   4.5 Synthetic exposure interventions (Table + Fig 6)
   4.closing Paragraph
5. Discussion                ← TODO
   5.1 Effective training pressure as organizing framework
   5.2 Relation to recursive-training literature
   5.3 Distributional degradation as diagnostic signal
   5.4 Practical implications for recursive training systems
6. Limitations               ← TODO (own section, ~1 page)
7. Conclusion                ← TODO (~0.5 page)
```

---

## Final Thesis

Recursive synthetic fine-tuning is governed by **effective training pressure**: a composite of update capacity, perturbation magnitude, and synthetic exposure. Degradation emerges when this pressure crosses a backbone-dependent boundary. The boundary is sharp: ~5% reduction in synthetic exposure shifts r=256 from degradative to near-homeostatic.

---

## Key Results (verified from JSONs)

### QLoRA Rank Dose-Response (Qwen 2.5 1.5B)
| Rank | Gen10 Ret | Eff Rank | Regime | Seeds |
|---|---|---|---|---|
| r=4 | 96.2% (Gen5) | 3.34 | Homeostatic | 1 |
| r=16 | 94.9% (N=3) | 11.08 | Homeostatic | 3 |
| r=32 | 94.9% (Gen5) | 17.85 | Homeostatic | 1 |
| r=64 | 91.1% | 29.52 | Homeostatic | 1 |
| r=128 | 88.6% | 50.16 | Bounded | 1 |
| r=256 | 78.0% (N=3) | 87.57 | Degradative | 3 |

### Cross-Backbone
- Gemma 3 1B: threshold eff rank ~3-9 (N=3 at r=4 and r=16)
- Gemma 3 post-threshold plateau: r=16 ≈ r=256 (~70%)
- Gemma 4 E2B: homeostatic at eff 2.13/5.64 (N=1)

### FFT vs QLoRA (N=3, Gen10)
- QLoRA r=16: 97.4% mean (K0=78) | FFT LR=1e-6: 91.9% mean | Gap: +5.6pp
- Gap is one-time adaptation cost (emerges Gen1, flat after)
- FFT fragile-fact floor: Jaccard=1.0 cross-seed (same 6 facts)
- QLoRA vs FFT: Jaccard=0.0 (different facts lost)

### Distributional Signatures
- Three-regime taxonomy: homeostatic / distributionally degraded / factually degradative
- r=128 dissociation: retention 88.6% bounded BUT content efficiency 0.13, persistence 5.2%
- Two phenotypes: r=128 filler verbosity vs r=256 elaborative drift
- Cross-lag: raw ρ=-0.965, partial ρ=-0.22 (co-symptom, not cause)

### Interventions (Qwen r=256, K0=78)
| Condition | Gen5 Ret | Delta |
|---|---|---|
| C1 Normal | 65/78 (83.3%) | — |
| C2 Short | 71/78 (91.0%) | +7.7pp |
| C3 Filtered | 72.0/78 mean (92.3%) | +9.0pp (N=3) |
| C4 Canonical | 69/78 (88.5%) | +5.1pp |
| C5 Random | 72.3/78 mean (92.7%) | +9.4pp (N=3 masks) |
- C3 vs C5 Jaccard=0.015 (different examples removed)
- C2 uses MORE tokens but still improves → not token-budget confound

---

## Critical Wording Rules

| SAY | DON'T SAY |
|---|---|
| effective training pressure | rank causes collapse |
| approximately perturbation-matched | exactly matched drift |
| output drift accompanies degradation | output drift causes degradation |
| pressure-gated at a sharp boundary | inevitable collapse above rank X |
| identifies the regime where predictions hold | contradicts Shumailov/Dohmatob |
| suggestive (different protocol) | controlled replication of Keisha |
| in the Qwen r=256 boundary case | any 5% universally works |
| diagnostic signature | early warning (not proven temporally) |
| over the observed 10-generation horizon | indefinitely |
| to our knowledge | first ever |
| is sufficient to produce | induces (too causal) |
| these results suggest/indicate | this confirms |
| consistent empirical separation | robust evidence |

---

## Writing Process Rules

1. **Never use em-dashes (---)** as punctuation. Always commas or periods.
2. **Write section here first**, show to user for GPT validation, then insert in .tex
3. **Each Results subsection ends with bridge paragraph** (2-3 sentences)
4. **Figures**: PDF vectorial in paper/figs/, generated by scripts/gen_fig*.py
5. **Style**: `science`, `ieee`, `no-latex` (SciencePlots)
6. **Compilation**: pdflatex in code/paper/, uses system TeX Live cas-dc.cls
7. **Statistics**: N=3 with per-seed + mean + range; N=1 labeled exploratory
8. **No overclaims**: every assertion checked against actual data in outputs/

---

## Figures Generated

| Fig | File | Content |
|---|---|---|
| 1 | fig1_trajectories.pdf | Longitudinal r=16/128/256 Gen0-10 |
| 2 | fig2_dose_response.pdf | Retention + eff rank vs nominal rank (2 panels) |
| 3 | fig3_cross_backbone.pdf | Qwen/Gemma3/Gemma4 (shapes=backbone, colors=regime) |
| 4 | fig4_fft_vs_qlora.pdf | FFT vs QLoRA Gen10, 3 seeds, gap annotated |
| 5 | fig5_distributional.pdf | Length/persistence/efficiency (3 panels) |
| 6 | fig6_interventions.pdf | C1-C5 retention + delta (2 panels) |

---

## References (in cas-refs.bib)

Core: shumailov2024, dohmatob2025, gerstgrasser2024, keisha2025, seddik2024, xu2025, zibakhsh2024, yi2025, adapala2025, alemohammad2024
PEFT: hu2022, dettmers2023, biderman2024, aghajanyan2021, roy2007
Data/metrics: joshi2017, li2016, mccarthy2010, holtzman2020, villalobos2024
Models: qwen2024, gemma2025
Other: dey2024

Full details in docs/REFERENCES_SEARCH.md

---

## Discussion Plan (validated by GPT)

### 5.1 Effective training pressure
- Three axes converge on same phenomenon
- Framework, not equation
- "Whether pressure is increased through capacity, magnitude, or exposure, the system responds qualitatively in the same way"

### 5.2 Relation to literature
- Shumailov/Dohmatob: we locate regime boundaries, not refute
- Keisha: suggestive context, different protocol
- Biderman: we map WHEN "forgets less" holds vs fails
- Gerstgrasser: orthogonal axis (data vs update)

### 5.3 Distributional degradation as diagnostic
- r=128 dissociation is key (retention ok, distribution degraded)
- Cross-lag weak after controls → co-symptom
- Practical: monitor length, efficiency, persistence, not just accuracy

### 5.4 Practical implications (connects to EAAI scope)
- Rank as control knob
- LR as control knob
- Exposure as control knob
- Monitoring signals: SDI, content efficiency, persistence
- Rule of thumb: if indicators degrade, reduce pressure

---

## Limitations Plan (Section 6, ~1 page)

1. Model scale: 1-2B only
2. Single dataset: TriviaQA factoid
3. N=3 for headline, N=1 for ablations
4. Gen10 maximum horizon
5. Cross-architecture normalization unsolved
6. Perturbation matching approximate
7. Intervention tested only at r=256 boundary
8. No weight-level mechanism analysis
9. Output metrics descriptive, not causal
10. Backbone generalization: only 2 thresholds located

---

## Conclusion Plan (Section 7, ~0.5 page)

1. Summary: pressure-gated recursive degradation
2. Main contributions (5 bullets)
3. Implication: boundary is sharp and controllable
4. Future work (1 paragraph): scale, datasets, causal decomposition, normalization

---

## File Locations

```
code/
├── paper/
│   ├── cas-dc-template.tex    ← MAIN FILE (13 pages, compiles)
│   ├── cas-refs.bib           ← 23 references
│   ├── model1-num-names.bst
│   ├── figs/                  ← 6 PDFs + 6 SVGs
│   └── .gitignore
├── docs/
│   ├── SESSION_STATE.md       ← THIS FILE
│   ├── PAPER_STRUCTURE.md     ← section/subsection plan
│   ├── PAPER_OUTLINE.md       ← figures/tables/wording guide
│   ├── PROJECT_STATUS_v2.md   ← all experimental results
│   ├── REFERENCES_SEARCH.md   ← 23 refs with DOIs/abstracts
│   ├── TEMPLATE_RULES.md      ← Elsevier CAS-DC rules
│   └── FUTURE_WORK.md         ← 18 future directions
├── scripts/
│   ├── gen_fig1.py through gen_fig6.py
│   └── [experiment scripts]
├── outputs/                   ← all JSON results
└── .kiro/rules/experiment-scripts.md  ← includes writing rules
```

---

## Hardware
- Local: Windows, RTX 3070 8GB (analysis only)
- Athena: RTX 4000 Ada 20GB (SSH, training done)
- TeX Live 2025 installed locally

## Key Technical Notes
- K0=79 for g1_rank* runs, K0=78 for fft_drift_gen10 and intervention runs
- Full-linear r=4 mean eff rank = 3.43 (confirmed)
- Qwen: 28L, 1536d, 12h, 2kv (confirmed from config.json)
- Gemma 3 1B: 26L, 1152d, 4h, 1kv (confirmed from config.json)
- Loss is over full prompt+response (DataCollatorForLanguageModeling mlm=False)
