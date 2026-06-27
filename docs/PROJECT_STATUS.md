# Project Status — Complete Summary

**Project:** Recursive Synthetic Fine-Tuning and Knowledge Degradation Under Low-Rank Adaptation  
**Researcher:** Julio Azancort  
**Date:** 2026-06-24  
**Repository:** https://github.com/Jazancort/llm-knowledge-collapse  

---

## 1. Origin

This experimental work extends the theoretical survey "Recursive Training Failures in Large Language Models: A Unified Taxonomy, Security Analysis, and Governance Framework" (Azancort, de Carvalho, Francês — submitted to Springer, 2026).

The survey identifies three collapse regimes (model collapse, knowledge collapse, directed collapse) but contains no original empirical validation. This project was designed to provide that validation.

---

## 2. Original Hypothesis (now evolved)

**Initial H1:** Representational instability (measured by CKA/ESI) precedes factual degradation under recursive synthetic training — acting as an early warning signal for Knowledge Collapse.

**What happened:** Knowledge Collapse did NOT occur in the regime tested (QLoRA/PEFT). Instead, a more nuanced and scientifically valuable discovery emerged.

---

## 3. Hardware & Setup

- **GPU:** NVIDIA RTX 3070 (8GB VRAM)
- **CPU:** Ryzen 5600X, 64GB RAM
- **Model:** Qwen2.5-1.5B-Instruct (4-bit quantized, QLoRA)
- **Dataset:** TriviaQA (2000 train, 200 eval, 100 probe)
- **Fine-tuning:** QLoRA, target_modules=[q_proj, v_proj], LR=1e-5, 2 epochs
- **Recursion protocol:** Data-only (new adapter from scratch each generation, base model frozen)
- **Evaluation:** Deterministic greedy decoding, exact match with alias normalization
- **Python:** 3.12, PyTorch 2.4.1+cu124, transformers 4.46.3

---

## 4. Experimental Timeline

| Phase | What was done | Key outcome |
|---|---|---|
| M0 | Model introspection | Decoder path confirmed, VRAM mapped (1.07GB used) |
| M1A | Probe extraction + CKA validation | Hooks work, CKA-Factual > CKA-Global confirmed |
| M1B | Single training cycle | Pipeline validated, no collapse in 1 cycle |
| M2 | 3 generations × 3 seeds (G1 synthetic) | Retention 94.5% ± 0.7%. No collapse. |
| G2-Control | 1 cycle with real data | CKA identical to G1 (~0.983). CKA doesn't distinguish data source. |
| G2-Full | 3 generations × 3 seeds (real data shards) | Real data maintains factual flux; synthetic freezes transitions. |
| G1-Gen10 | 10 generations, r=16, seed 15 | Homeostasis confirmed (94.9% stable, no degradation) |
| Rank ablation | r=4, r=16, r=64, r=128(10gen), r=256(10gen×3seeds) | **CENTRAL DISCOVERY** |

---

## 5. Central Discovery: Capacity-Gated Regime Transition

### Core finding:

Recursive synthetic fine-tuning under QLoRA exhibits a capacity-gated transition between bounded retention and progressive degradation. This transition is architecture-general but the critical threshold is backbone-dependent.

**Bounded/Homeostatic Regime (below backbone-specific threshold):**
- Retention remains stable (88-97%) with no runaway degradation over 10 generations
- System absorbs perturbation without cumulative loss
- Effective rank remains stable across generations

**Degradative Regime (above backbone-specific threshold):**
- Retention declines progressively
- In Qwen: ~2.4 pp/gen, reaching 78.0% ± 2.6% at Gen10
- In Gemma: ~6 pp/gen, reaching 68.8% ± 1.2% at Gen5
- Plasticity depletes (W→C → 0 in later generations)

### Threshold comparison (multi-seed confirmed):

| Backbone | Homeostatic | Degradative | Threshold (mean eff rank) | Seeds |
|---|---|---|---|---|
| Qwen 2.5 1.5B | r≤128, eff≤50 (94.9% ± 0.7%) | r=256, eff~88 (78.0% ± 2.6%) | ~50–88 | 3 |
| Gemma 3 1B IT | r≤4, eff≤3 (92.2% ± 1.2%) | r≥16, eff≥9 (68.8% ± 1.2%) | ~3–9 | 3 |
| Gemma 4 E2B IT | r≤16, eff≤5.6 (97.4%) | not located | unknown (>5.6) | 1 |

### Critical: NO universal effective-rank threshold exists

Qwen is homeostatic at eff rank 11, 18, 30, 50. Gemma 3 is degradative at eff rank 9.3. Same raw effective rank, opposite regimes, different backbones. Therefore:

- **Within a backbone:** effective rank cleanly predicts regime (Qwen dose-response is monotonic)
- **Across backbones:** raw effective rank is NOT commensurable. The threshold differs by ~10× (Qwen ~50-88 vs Gemma 3 ~3-9)
- **Why:** architectures differ in hidden dim, depth, GQA head ratio — "eff rank 9" means different functional perturbation magnitudes in each model

**Defensible claim:** Regime structure is architecture-general; raw effective-rank threshold is backbone-dependent. Normalizing capacity across architectures is future work.

**Gemma 4 E2B IT (n=1, single-seed robustness case):**
- r=4: eff 2.1, 96.1% — homeostatic
- r=16: eff 5.6, 97.4% — homeostatic (low eff rank, not testing threshold)
- No degradative point found — threshold not located
- Role in paper: additional backbone confirming low-eff-rank stability, not a headline claim

### Dose-Response Curve (5 data points):

| Rank | Trainable Params | Gen 5 Retention | Gen 10 Retention | Effective Rank | % Utilization |
|---|---|---|---|---|---|
| r=4 | 545K (0.06%) | 96.2% | — | 3.34 | 84% |
| r=16 | 2.2M (0.24%) | 94.9% | 94.9% | 11.08 | 69% |
| r=64 | 8.7M (0.98%) | 89.9% | — | 30.08 | 47% |
| r=128 | 17.4M (1.96%) | 87.3% | 88.6% | 50.5 | 39% |
| r=256 | 34.9M (3.91%) | 83.1% | 78.0% ± 2.6% | 87.8 | 34% |

### Key observations:

1. **Monotonic dose-response:** More adapter capacity → more factual loss (but NOT linear)
2. **Sublinear effective rank growth:** Doubling nominal rank does NOT double effective rank
3. **Diminishing utilization:** The model recruits proportionally LESS capacity as rank increases
4. **Regime transition:** Between r=128 (bounded) and r=256 (degrading) — exact location undetermined
5. **Degradation under constant complexity:** r=256 eff_rank stays ~87 while retention falls — degradation is not driven by increasing adapter complexity

### Sprint 1 Finding — Module Targeting Invariance:

Adapting all linear layers (Full Linear: q,k,v,o,gate,up,down) vs attention-only (q,v) does NOT change the behavioral regime. Full Linear r=4 (4.6M params) retains 94.9% — identical to Attention r=16 (2.2M params). Full Linear r=16 at Gen10 = 87.3%, comparable to Attention r=128 at Gen10 = 88.6%. The difference is within single-seed noise.

Mean effective rank per adapter is the primary predictor of retention, though not an exclusive one. Module allocation may modulate retention as a secondary, underpowered observation.

---

## 6. Secondary Findings

### CKA-Factual vs CKA-Global
- CKA-Factual (last prompt token) detects representational adaptation that CKA-Global (mean pool) misses
- However, CKA does NOT distinguish synthetic from real data (both produce ~0.983 drop)
- CKA measures adaptation intensity, not data source toxicity

### G1 (Synthetic) vs G2 (Real) — Transition Dynamics
- CKA impact is identical between groups
- The DIFFERENCE is in factual transitions: G1 freezes (0 transitions after Gen 1), G2 maintains flux (2-7 transitions/gen)
- Under r=16, synthetic recursion produces "factual ossification" rather than collapse

### Adapter Health
- Effective rank remains stable across all regimes (no rank collapse observed)
- No evidence of geometric degeneration of LoRA matrices
- The earlier theoretical concern about adapter pathology was unfounded

### Accuracy Audit
- TriviaQA baseline accuracy: 39.5% (model genuinely doesn't know 54.5% of answers)
- 0 formatting errors — exact match is NOT the problem
- Retention Accuracy (K0 subset) is the valid metric for measuring collapse

---

## 7. What Was Refuted

| Hypothesis | Status | Evidence |
|---|---|---|
| Knowledge Collapse is inevitable under synthetic recursion | **REFUTED** for r≤128 | 10 gens, 3 seeds, 94.9% retention |
| Stage B (valley of dangerous competence) | **NOT OBSERVED** | Model became LESS confident, not more |
| CKA detects recursion-specific damage | **REFUTED** | G2 real data produces same CKA drop |
| Adapter rank collapse causes degradation | **REFUTED** | Effective rank stable in all regimes |
| Scaling laws of collapse apply to PEFT | **PARTIALLY REFUTED** | Only manifests above capacity threshold |

---

## 8. What Was Confirmed

| Finding | Evidence |
|---|---|
| Recursive synthetic training CAN cause progressive degradation | r=256, 3 seeds, 78% at Gen10 |
| The degradation depends on adapter capacity, not just data | Dose-response curve with 5 points |
| Low-rank adaptation provides implicit regularization | r=4 through r=128 all stabilize |
| There exists a capacity threshold for the protective mechanism | Between r=128 (stable) and r=256 (degrading) |
| CKA-Factual is more sensitive than CKA-Global | Confirmed with real training, not just perturbation |
| The base model's frozen weights resist corruption | Even r=256 doesn't collapse catastrophically |

---

## 9. Relationship to Literature

| Paper | Claim | Our finding |
|---|---|---|
| Shumailov 2024 (Nature) | Recursive training → irreversible collapse | Not under PEFT (r≤128). Consistent with their result at r=256. |
| Dohmatob 2025 (ICLR) | Any k>0 synthetic → linear error growth | PEFT alters the effective hypothesis class; their high-capacity assumptions don't directly apply at low rank. r=256 shows ~linear growth consistent with their bounds when capacity is high. |
| Keisha 2025 | Three-stage knowledge collapse (Gemma 3 1B, FFT) | Stage B not observed under QLoRA. Different update regime. |
| Gerstgrasser 2024 | Accumulation prevents collapse | Low-rank PEFT also bounds collapse under replace protocol — complementary mechanism. |
| Xu 2025 | P(improvement) < 1/2 under recursion | Consistent: homeostasis = no improvement, but no collapse either. |
| Biderman 2024 (TMLR) | LoRA learns less and forgets less than FFT | Our results are consistent. We extend by showing the dose-response: HOW MUCH rank determines WHETHER forgetting accumulates. |
| Adapala 2025 | Anti-Ouroboros: Gemma + cumulative LoRA + 5 gens | Different protocol (cumulative + selection filter). We use replace-without-filter and map the capacity threshold. |
| Zibakhsh 2024 | TCE loss delays collapse 2.3× | LoRA and TCE are orthogonal mitigation strategies: TCE modifies the loss signal, LoRA restricts the update subspace. |

**Positioning:** We do NOT claim novelty of "using LoRA for recursive training" — that exists (Biderman, Adapala). Our contribution is the **systematic mapping of adapter capacity as the control variable**, demonstrating a dose-response relationship and identifying the regime transition. This is mechanistic refinement of a known protective effect, not discovery of the effect itself.

---

## 10. Proposed Paper Title

**"Capacity-Dependent Knowledge Degradation Under Recursive Synthetic Fine-Tuning: A Dose-Response Analysis of Low-Rank Adaptation"**

Or shorter:

**"How Much Rank Is Too Much? Mapping the Capacity Threshold for Knowledge Degradation Under Recursive Synthetic LoRA"**

---

## 11. Proposed Paper Structure

1. **Introduction:** Recursive training crisis (from survey), gap in empirical PEFT studies
2. **Related Work:** Shumailov, Dohmatob, Keisha, Gerstgrasser — all study FFT regime
3. **Methodology:** QLoRA protocol, data-only recursion, K0 retention metric, rank ablation design
4. **Results:**
   - 4.1: No collapse at r≤128 (3 seeds, 10 gens)
   - 4.2: Progressive degradation at r=256 (3 seeds, 10 gens)
   - 4.3: Dose-response curve (r=4 through r=256)
   - 4.4: Effective rank saturation and utilization curve
   - 4.5: CKA-Factual as adaptation instrument
   - 4.6: G1 vs G2 (synthetic freezes, real maintains flux)
5. **Discussion:** Noise conduction threshold, reconciliation with theory, practical implications
6. **Limitations:** Single model (1.5B), single dataset, no FFT comparison, threshold not precisely located
7. **Conclusion:** Adapter rank as governance parameter for recursive training safety

---

## 12. Key Figures for Paper

1. **Longitudinal trajectories (Fig 1):** r=16 vs r=128 vs r=256 retention over 10 generations (with ±std band for r=256)
2. **Dose-response curve (Fig 2):** Factual loss and effective rank vs nominal rank (dual axis, log₂ scale)
3. **Plasticity depletion (Fig 3):** C→W vs W→C transitions per generation, r=128 vs r=256 (bar chart)
4. **Utilization curve (Fig 4):** % effective/nominal rank showing diminishing returns
5. **Retention vs Mean Effective Rank — Gen 5 (Fig 5a):** All configs (attention + full-linear), showing trend
6. **Retention vs Mean Effective Rank — Gen 10 (Fig 5b):** Configs with Gen10 data, marking degradative zone

---

## 13. Open Questions (Future Work)

1. Where exactly is the threshold? (r=192 experiment not yet run)
2. Is the transition abrupt or gradual?
3. Does the pattern hold for larger models (3B, 7B)?
4. Does Full Fine-Tuning produce collapse at r=16-equivalent parameter counts?
5. Does domain anchoring (Keisha's mitigation) interact with rank?
6. What is the relationship between dataset entropy and effective rank saturation?

---

## 14. Files in Repository

```
code/
├── scripts/
│   ├── m0_introspect.py          # Model architecture mapping
│   ├── m1a_extract.py            # Probe extraction validation
│   ├── m1a_cka_tests.py          # CKA identity + sensitivity
│   ├── m1a_baseline.py           # Accuracy baseline (corrected: chat template + entropy)
│   ├── m1b_single_cycle.py       # Single QLoRA cycle
│   ├── m2_pilot.py               # 3 gen recursive (with checkpoints)
│   ├── m2_multiseed.py           # Multi-seed replication
│   ├── m2_audit.py               # Accuracy error classification
│   ├── m2_retention.py           # K0 retention + transition matrix
│   ├── g2_control.py             # Real data CKA baseline
│   ├── g2_full.py                # 3 gen real data (multi-seed)
│   ├── g1_gen10.py               # 10 generation extension
│   └── g1_rank_ablation.py       # Rank ablation (4/16/64/128/256)
├── docs/
│   ├── PROTOCOL.md               # Full experimental protocol
│   ├── DECISIONS.md              # 15 design decisions (D01-D15)
│   ├── METRICS.md                # Formal metric definitions
│   ├── SCENARIOS.md              # Pre-defined result interpretation
│   ├── THEORY.md                 # Connection to survey
│   ├── CHECKLIST.md              # Operational checklist
│   ├── GRILL_SESSION.md          # First grill (10 questions)
│   ├── GRILL_SESSION_2.md        # Second grill (D16-D58, engineering)
│   └── RESULTS_FINAL.md          # Initial results synthesis
├── outputs/                       # All experimental data (JSON)
├── configs/                       # YAML configurations
└── src/                           # Reusable modules
```

---

## 15. Statistical Summary (paper-ready)

### Qwen 2.5 1.5B — Headline claims (N=3 seeds):
- **Homeostatic (r=16):** 94.9% ± 0.7% at Gen10
- **Degradative (r=256):** 78.0% ± 2.6% at Gen10
- Difference: 16.9 pp, non-overlapping bands
- Rate of degradation (r=256): ~2.4 pp/generation

### Gemma 3 1B IT — Headline claims (N=3 seeds):
- **Homeostatic (r=4):** 92.2% ± 1.2% at Gen5; 93.6% at Gen10 (seed 15)
- **Degradative (r=16):** 68.8% ± 1.2% at Gen5
- Difference: 23.4 pp, non-overlapping bands
- Degradative plateau: r=16 and r=256 both reach ~70% (capacity beyond threshold doesn't worsen)

### Qwen supporting points (N=1, exploratory):
- r=4: 96.2% Gen5, r=32: 94.9% Gen5, r=64: 91.1% Gen10, r=128: 88.6% Gen10
- Full Linear r=4: 94.9% Gen5, Full Linear r=16: 87.3% Gen10

### Gemma supporting points (N=1):
- r=2: 97.9% Gen5 (strongly homeostatic)
- r=256: 70.2% Gen5 (same as r=16 — plateau effect)

### Jaccard analysis (Gemma r=16 vs r=256 lost facts):
- Gen3: 0.571, Gen5: 0.474 (declining overlap → diverging paths)
- Interpretation: mixed floor + path-dependent degradation

### Cross-architecture:
- Threshold differs ~10× (eff rank ~50-88 in Qwen vs ~3-9 in Gemma)
- Regime structure generalizes; threshold is backbone-dependent
- Post-threshold behavior differs: Qwen continues declining; Gemma plateaus
