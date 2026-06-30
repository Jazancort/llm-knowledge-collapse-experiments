# SESSION STATE — Full Consolidated (2026-06-30 18:30)

## CRITICAL: Read this file after any context compaction.
## Contains everything needed to continue working on the paper.

---

## Project Overview

**Paper:** "Effective Training Pressure Gates Recursive Knowledge Degradation in LLMs: A Multi-Axis Dose-Response Study"
**Target:** Engineering Applications of Artificial Intelligence (Elsevier, IF 9.0, A1 CAPES)
**Researcher:** Julio Azancort (UFPA)
**Repo:** https://github.com/Jazancort/llm-knowledge-collapse
**Latest commit:** af4ac9e (2026-06-30)
**Main file:** code/paper/cas-dc-template.tex (20 pages double-column, compiles OK)
**Bib:** code/paper/cas-refs.bib (44 references cited)

---

## PAPER STATUS: COMPLETE FIRST DRAFT

ALL sections written, validated, and compiled:

| Section | Pages (est.) | Status |
|---|---|---|
| Abstract | ~0.3 | ✅ (4 findings, ~210 words) |
| Highlights | 5 items | ✅ (experimental, no overclaims) |
| Keywords | 6 items | ✅ |
| 1. Introduction | ~2 | ✅ (7 paragraphs + contributions list) |
| 2. Related Work (2.1-2.5) | ~2 | ✅ (5 subsections, 44 refs) |
| 3. Methodology (3.1-3.8) | ~4 | ✅ (8 subsections + Fig 7 methodology) |
| 4. Results (4.1-4.5) | ~5 | ✅ (5 subsections, 5 tables, 5 data figs) |
| 5. Discussion (5.1-5.4) | ~2.5 | ✅ (4 subsections) |
| 6. Limitations | ~1.5 | ✅ (10 items, fortified against 5 reviewer attacks) |
| 7. Conclusion | ~1 | ✅ (4 paragraphs + future work) |
| **Total** | **20 pages** | ✅ |

---

## Figures (7 total)

| Fig | File | Content | Location |
|---|---|---|---|
| 1 | fig1_trajectories.png + fig2_dose_response.png | Combined: trajectories + dose-response | §4.1 |
| 2 | fig3_cross_backbone.png | Retention vs eff rank, 3 backbones | §4.2 |
| 3 | fig4_fft_vs_qlora.png | FFT vs QLoRA Gen10, 3 seeds | §4.3 |
| 4 | fig5_distributional.png | Length/persistence/efficiency 3 panels | §4.4 |
| 5 | fig6_interventions.png | C1-C5 retention + delta, 2 panels | §4.5 |
| 6 | fig7_methodology.png | Vertical timeline pipeline diagram | §3 (before 3.4) |

Note: Fig 1+2 are combined in single captionof. Fig 6 is figure* (full-width).
All figures are non-float (captionof) except fig6 and fig7.

---

## Tables (5 total)

| Table | Content | Location |
|---|---|---|
| 1 | Experimental axes | §3.4 |
| 2 | QLoRA rank sweep (Qwen) | §4.1 |
| 3 | Gemma 3 results | §4.2 |
| 4 | FFT LR sweep | §4.3 |
| 5 | Interventions C1-C5 | §4.5 |

---

## Key Design Decisions (FINAL)

- **No \paragraph{}** — zero in entire paper. Only section/subsection/subsubsection.
- **6 subsubsections** total: Effective rank, Perturbation magnitude, QLoRA vs FFT, Homeostatic, Bounded, Degradative.
- **Non-float figures** (captionof) — fixes cas-dc.cls float placement issues.
- **Numbered citations** (model1-num-names.bst).
- **Section~\ref** (not \S).
- **No em-dashes** as punctuation.
- **Figures as PNG 300dpi** (not PDF — was broken rendering).
- **Fig 7 methodology** generated from HTML via Playwright (vertical timeline).

---

## Compilation

```bash
cd "G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\code\paper"
pdflatex -interaction=nonstopmode cas-dc-template.tex
bibtex cas-dc-template
pdflatex -interaction=nonstopmode cas-dc-template.tex
pdflatex -interaction=nonstopmode cas-dc-template.tex
```

Full sequence needed for all refs/cites. Uses system TeX Live 2025.

---

## Writing Rules (CRITICAL)

1. NEVER use em-dashes (---) as punctuation. Always commas or periods.
2. "suggests/indicates" not "confirms"
3. "approximately perturbation-matched" not "exactly matched"
4. "in the Qwen r=256 boundary case" not "any 5% universally works"
5. "identifies regime boundaries" not "contradicts prior work"
6. "diagnostic signature" not "early warning" (temporal precedence not proven)
7. "We refer to this as..." (our terms) not "This represents..." (established)
8. Cross-lag: "co-symptom" not "causal antecedent"
9. No overclaims: zero-cost policy enforced throughout

---

## Key Improvements Applied (2026-06-30)

### From comparative analysis of 7 base articles:
- **Abstract**: r=128 "threefold" + "above 88%"; intervention "approximately 5%" explicit
- **Introduction**: "requires no external verifier, no additional real data, no changes to training objective"
- **Highlights**: "order of magnitude" explicit
- **Discussion 5.2**: "practical response to theoretical pessimism of prior work"
- **Discussion 5.4**: zero-cost governance repeated

### Reviewer fortification (5 attacks defended in Limitations):
1. Toy model: justified by prior work scale + TriviaQA isolates factual retention
2. No math formula: phenomenological mapping precedes formal modeling
3. QLoRA quantization: base static, adapters in higher precision, 16-bit comparison future work
4. Boundary cherry-pick: deliberate (demonstrates sharpness)
5. Format collapse K0: C2 proves actual knowledge loss, not just format

### Reference fixes:
- Contradiction [41] vs "open question" → resolved (suavized wording)
- Preprints [27][28][29] "confirm" → "suggest"
- Yi arXiv ID completed (2510.16657)
- Guo repositioned from anchor to support
- ROME (Meng 2022) added for module topology context

---

## Central Thesis

Recursive synthetic fine-tuning stability is governed by **effective training pressure** (adapter capacity × update magnitude × synthetic exposure). There is a **sharp backbone-dependent threshold**: below it, homeostatic; above it, factual degradation with distributional signatures. The boundary is controllable: ~5% exposure reduction shifts r=256 from degradative to near-homeostatic.

---

## Key Data (verified from JSONs)

### QLoRA Rank (Qwen 2.5 1.5B, K0=79)
| Rank | Gen10 Ret | Eff Rank | Regime |
|---|---|---|---|
| r=4 | 96.2% (Gen5) | 3.34 | Homeostatic |
| r=16 | 97.4% (N=3) | 11.08 | Homeostatic |
| r=64 | 91.1% | 29.52 | Homeostatic |
| r=128 | 88.6% | 50.16 | Bounded |
| r=256 | 78.0% ±2.6% (N=3) | 87.57 | Degradative |

### Cross-Backbone (Gemma 3, K0=47)
- r=4: 92.9% mean (N=3) | r=16: 68.8% mean (N=3)
- Threshold eff rank ~3-9 (10× lower than Qwen)

### FFT vs QLoRA (K0=78, Gen10, N=3)
- QLoRA r=16: 97.4% | FFT LR=1e-6: 91.9% | Gap: +5.6pp

### Interventions (Qwen r=256, K0=78, Gen5)
- C1: 65/78 (83.3%) | C3: 72.0/78 mean (92.3%) | C5: 72.3/78 mean (92.7%)
- Jaccard C3 vs C5: 0.015 (different examples removed)

### r=128 Dissociation
- Retention: 88.6% (bounded) BUT content efficiency: 0.13, persistence: 5.2%, MTLD: 745

---

## File Locations

```
code/
├── paper/
│   ├── cas-dc-template.tex    ← MAIN FILE (20 pages)
│   ├── cas-refs.bib           ← 44 references
│   ├── model1-num-names.bst
│   ├── figs/                  ← 7 PNGs + PDFs + SVGs
│   └── .gitignore
├── docs/
│   ├── SESSION_STATE.md       ← THIS FILE
│   ├── COMPARATIVE_ANALYSIS.md ← 7-item analysis of base articles
│   ├── PROJECT_STATUS_v2.md   ← all experimental results
│   ├── PAPER_OUTLINE.md       ← wording guide
│   ├── REFERENCES_SEARCH.md   ← ref search results
│   └── [other docs]
├── scripts/
│   ├── gen_fig1.py through gen_fig7.py
│   ├── gen_all_png.py
│   ├── fig7_methodology.html  ← source for Fig 7
│   └── [experiment scripts]
├── outputs/                   ← all JSON results
└── .kiro/rules/
```

---

## What's Next (possible tasks)

1. **Final consistency check** — verify all \ref, \cite, numbers match across sections
2. **Cover letter** for submission
3. **Data availability statement** (GitHub repo link)
4. **CRediT author statement**
5. **Supplementary material** (extended tables, per-seed data)
6. **Proofreading pass** — check for typos, grammar, flow
7. **Apply remaining insights from COMPARATIVE_ANALYSIS.md** if desired
8. **Update SESSION_STATE** before submission

---

## Technical Notes

- K0=79 for g1_rank* runs (Qwen main sweep)
- K0=78 for fft_drift_gen10 and all intervention runs
- K0=47 for Gemma 3 runs
- Qwen: 28L, 1536d, 12h, 2kv
- Gemma 3: 26L, 1152d, 4h, 1kv
- Training: 2 epochs, LR 1e-5, batch 4, grad_accum 2
- Generation: T=0.7, max_new_tokens=50
- Seeds: 15, 137, 256
- Playwright installed for HTML→PNG conversion (fig7)
