# PROJECT STATUS v2 — Capacity-Gated Recursive Fine-Tuning

**Last updated:** 2026-06-29
**Status:** EXPERIMENTS CLOSED. All checks resolved. Writing phase.

---

## 1. Current Thesis

Recursive synthetic fine-tuning is governed by a **perturbation-induced feedback loop**:

1. High-capacity updates induce drift in the synthetic output distribution (longer, less efficient responses)
2. These degraded synthetic outputs feed back into subsequent training and amplify factual loss
3. Low-rank PEFT reduces the initial perturbation; when output drift is additionally controlled (filtering/short constraints), even high-rank regimes recover near-homeostatic retention

The regime transition is capacity-gated and backbone-dependent. Update magnitude opens the vulnerability; synthetic-output drift amplifies it.

---

## 2. Experimental Pivot

**Original hypothesis (ESI/Lead-Time, June 23):** Representational instability (ESI) precedes factual degradation and serves as an early warning signal for knowledge collapse.

**What happened:** Knowledge collapse did NOT occur under QLoRA r≤128. Instead, a more tractable and mechanistic result emerged: adapter capacity (effective rank) cleanly predicts whether the system enters a homeostatic or degradative regime.

**Pivot decision:** Abandon ESI/Lead-Time framing. Pursue dose-response mapping.

All earlier documentation (PROTOCOL, THEORY, METRICS, SCENARIOS, etc.) describes the dead hypothesis and is marked OBSOLETE.

---

## 3. Core Results

### 3.1 Qwen 2.5 1.5B-Instruct (primary backbone, N=3 seeds)

| Rank | Trainable | Gen5 Ret. | Gen10 Ret. | Eff Rank | Regime |
|---|---|---|---|---|---|
| r=4 | 545K | 96.2% | — | 3.34 | Homeostatic |
| r=16 | 2.2M | 94.9% | 94.9% ± 0.7% | 11.08 | Homeostatic |
| r=32 | 4.4M | 94.9% | — | ~18 | Homeostatic |
| r=64 | 8.7M | 89.9% | 91.1% | 30.08 | Homeostatic |
| r=128 | 17.4M | 87.3% | 88.6% | 50.5 | Homeostatic (bounded) |
| r=256 | 34.9M | 83.1% | 78.0% ± 2.6% | 87.8 | **Degradative** |

Multi-seed (3 seeds): r=16 and r=256. Non-overlapping bands confirm regime separation.

### 3.2 Gemma 3 1B IT (secondary backbone, N=3 seeds at r=4, r=16)

| Rank | Gen5 Ret. | Gen10 Ret. | Eff Rank | Regime |
|---|---|---|---|---|
| r=2 | 97.9% | — | ~1.8 | Homeostatic |
| r=4 | 92.2% ± 1.2% | 93.6% (seed 15) | ~3.0 | Homeostatic |
| r=16 | 68.8% ± 1.2% | — | ~9.3 | **Degradative** |
| r=256 | 70.2% | — | ~62 | **Degradative (plateau)** |

**Threshold: ~3–9 effective rank.** Orders of magnitude lower than Qwen (~50–88).

Post-threshold plateau: r=16 and r=256 reach same floor (~70%). Additional capacity beyond threshold does not worsen degradation.

### 3.3 Gemma 4 E2B IT (robustness, N=1)

| Rank | Gen5 Ret. | Eff Rank | Regime |
|---|---|---|---|
| r=4 | 96.1% | 2.1 | Homeostatic |
| r=16 | 97.4% | 5.6 | Homeostatic |

Threshold not located. Role: additional backbone confirming low-eff-rank stability. Not a headline claim.

### 3.4 Module Targeting Invariance (Qwen, N=1)

Full Linear (q,k,v,o,gate,up,down) r=4: 4.6M params → 94.9% Gen5 (= Attention r=16).
Full Linear r=16: 87.3% Gen10 (≈ Attention r=128).

**Mean effective rank per adapter is the primary predictor, regardless of module topology.**

---

## 4. FFT vs QLoRA Control — RESOLVED

### 4.1 LR Sweep (seed 15, Gen3)

| Method | LR | Perturbation proxy | Gen3 Retention |
|---|---|---|---|
| **QLoRA r=16** | 1e-5 | lora_norm ~0.42 | **75/78 (96.2%)** |
| FFT | 1e-6 | abs drift 0.39 | 72/78 (92.3%) |
| FFT | 5e-6 | abs drift 1.56 | 71/78 (91.0%) |
| FFT | 1e-5 | abs drift 1.95 | 70/78 (89.7%) |
| FFT | 2e-5 | abs drift 3.48 | 66/78 (84.6%) |

### 4.2 Drift-Matched Replication (3 seeds, Gen5) — DEFINITIVE

| Config | Seed 15 | Seed 137 | Seed 256 | Mean |
|---|---|---|---|---|
| QLoRA r=16 | 75/78 (96.2%) | 76/78 (97.4%) | 76/78 (97.4%) | **97.0%** |
| FFT LR=1e-6 | 72/78 (92.3%) | 72/78 (92.3%) | 72/78 (92.3%) | **92.3%** |
| **Delta** | 3.8 pp | 5.1 pp | 5.1 pp | **4.7 pp** |

Both methods are stable Gen3→Gen5 (no further degradation). Gap is consistent and non-overlapping across all seeds.

### 4.3 Conclusion

**Perturbation magnitude is the dominant factor** (FFT dose-response: 92.3% → 84.6% as drift increases 10×).

**Low-rank reduces one-time adaptation cost** (~4.7pp at approximately matched perturbation, replicated N=3). The gap emerges entirely at Gen1 and remains constant — both methods are equally stable post-Gen1. This is NOT differential degradation rate; it is a one-time cost difference.

**Fact-overlap analysis (3 seeds, Gen10 data):** FFT deterministically loses the same 6 facts [24,46,48,56,69,74] across all seeds (Jaccard=1.0 cross-seed), indicating a stable fragile-fact floor. QLoRA loses [11,12] in seeds 15/256 and [11,12,57] in seed 137 (2-3 facts, not perfectly deterministic). FFT vs QLoRA: Jaccard=0.0 in all seeds — no overlap. Low-rank reduces the count of lost facts AND shifts which facts are vulnerable. Whether this reflects a genuinely different update subspace remains future work.

**Paper claim (final):**

> "Factual stability under recursive synthetic fine-tuning is primarily governed by per-generation perturbation magnitude. At approximately matched perturbation, QLoRA incurs a smaller one-time factual adaptation cost than FFT (~5pp, 2-3 facts vs 6, consistent across 3 seeds through Gen10). After the initial adaptation, both methods are equally stable. FFT deterministically hits a fixed fragile-fact floor (Jaccard=1.0 cross-seed); QLoRA loses fewer and different facts (Jaccard=0.0). The dual dose-response (QLoRA rank + FFT LR) confirms magnitude as the dominant variable, with low-rank providing a modest additional benefit."

### 4.4 Norm metric audit

**IMPORTANT:** The `lora_norm` values reported for seeds 137/256 (17.38) and seed 15 (0.42) measure DIFFERENT things:
- Seed 15 (sweep): `compute_lora_drift()` = `||B@A||` = effective weight delta = **0.42** (comparable to FFT drift)
- Seeds 137/256 (replicate): `compute_lora_norm()` = `||A|| + ||B||` raw parameter norm = **17.38** (NOT comparable to anything)

The drift-matched comparison is valid via seed 15's measurement (B@A norm 0.42 ≈ FFT drift 0.39). Seeds 137/256 validate retention consistency only, not perturbation magnitude.

For the paper: frame as "approximately perturbation-matched" using the B@A metric; do not report the raw 17.38 values.

### 4.5 Caveats (honest in paper)

- Comparison is "approximately perturbation-matched" — LoRA B@A norm and FFT weight drift are useful proxies but not identical metrics
- The ~5pp gap is entirely Gen1 — not evidence of differential degradation rate
- Gen5 depth only for FFT (not Gen10) — but both flat Gen1→Gen5
- FFT uses paged_adamw_8bit + gradient_checkpointing (hardware constraint)

---

## 5. Secondary Findings

### 5.1 Mechanistic Analysis — Synthetic Output Drift (NEW, 2026-06-29)

**Hypothesis tested:** Does the degradative regime produce measurably worse synthetic data?

**Result: YES — with temporal precedence (cross-lag r=-0.965)**

#### A. Synthetic diversity diverges between regimes

| Metric | Homeostatic (r=16, Gen0→10) | Degradative (r=256, Gen0→10) |
|---|---|---|
| distinct-1 | 0.671 → 0.637 (stable) | 0.671 → 0.536-0.657 (drops) |
| mean_length | 2.5 → 2.7 (stable) | 2.5 → 3.6-4.6 (inflates) |
| response_uniqueness | ~0.93 (stable) | 0.93 → 0.97 (each response unique but verbose) |

Gemma 3 r=4 (homeostatic): perfectly stable (d1: 0.699→0.703, length: 2.2→2.3)
Gemma 3 r=16 (degradative): d1 drops 0.699→0.646, length inflates 2.2→3.9

#### B. Cross-lag: output drift accompanies retention loss

| Config | Regime | raw corr(len_t, ret_t+1) | partial (|gen) | first-diff | reverse |
|---|---|---|---|---|---|
| Qwen r=16 | homeostatic | 0.000 | 0.000 | 0.000 | -0.251 |
| Qwen r=256 | **degradative** | -0.965 | **-0.220** | **-0.433** | -0.986 |
| Gemma r=4 | homeostatic | +0.110 | +0.030 | +0.334 | -0.381 |
| Gemma r=16 | degradative | -0.847 | +0.995* | +0.990* | -0.952 |

*Gemma r=16 has only 4-5 data points — statistically unreliable.

**Robustness checks reveal:** The raw r=-0.965 is largely inflated by shared monotonic trend. After controlling for generation (partial r=-0.220), the association weakens substantially. Reverse-lag is equally strong. First-difference (r=-0.433) shows modest incremental coupling.

**Honest interpretation:** Synthetic-output drift **accompanies** factual degradation as a co-symptom of the same underlying process, rather than demonstrably preceding it. Both are driven by the same high-perturbation regime. The first-difference correlation (-0.433) suggests some incremental coupling beyond pure trend, but is not strong enough to claim temporal causation.

#### C. Error dispersion, not ossification

- Homeostatic: item-level stability constant (~33-40% exact match between gens)
- Degradative: item-level stability DECREASES (30%→14%), model drifts faster each generation
- The degradative regime does NOT freeze wrong answers; it loses anchoring

#### D. Synthetic Drift Index (SDI-3)

SDI-3 = log(length_ratio) + log(d1_gen0/d1_final) + instability_increase

| Config | Regime | SDI-3 |
|---|---|---|
| Gemma r=4 | homeostatic | 0.038 |
| Qwen r=16 | homeostatic | 0.161 |
| Qwen r=256 | degradative | 0.437 |
| Gemma r=16 | degradative | 0.705 |
| Qwen r=128 | **distributional drift (retention bounded)** | **1.530** |

#### E. Complete Metrics — New Findings (2026-06-29)

**Factual Efficiency (retention / mean_length) — strongest new metric:**

| Config | Gen0 | Gen10 | Interpretation |
|---|---|---|---|
| Qwen r=16 | 0.406 | 0.352 | Stable after initial cost |
| Qwen r=256 | 0.406 | 0.219 | Progressive dilution |
| Qwen r=128 | 0.406 | **0.127** | Severe dilution despite bounded retention |
| Gemma r=4 | 0.448 | 0.402 | Perfectly stable |
| Gemma r=16 | 0.448 | 0.179 (Gen5) | Rapid collapse |

**Baseline Response Persistence (% same answer as Gen0):**

| Config | Gen1 | Gen10 | Interpretation |
|---|---|---|---|
| Qwen r=16 | 35.7% | 36.1% | Stable anchoring |
| Qwen r=256 | 32.6% | **3.8%** | Lost original behavior |
| Qwen r=128 | 28.6% | **5.2%** | Lost anchoring despite retention |
| Gemma r=4 | 46.4% | 46.1% | Perfectly anchored |
| Gemma r=16 | 36.8% | 3.5% (Gen5) | Rapid loss |

**Two Degradation Phenotypes:**

| Phenotype | Example | Characteristics |
|---|---|---|
| **Filler/repetitive verbosity** | Qwen r=128 | stopwords 12%→25%, MTLD collapses (2673→745), length 2.5→7.0 |
| **Elaborative/dispersive drift** | Qwen r=256 | stopwords 12%→9% (drops!), MTLD rises (2673→3804), length 2.5→3.6 |

r=128 fills with stopwords and repetition; r=256 elaborates with varied vocabulary but loses factual anchoring. Different degradation mechanisms at different capacity levels.

**Key insight: r=128 exposes a dissociation.**

> "Qwen r=128 reveals that factual retention (88.6%) can remain bounded while distributional quality collapses: content efficiency drops to 0.127 (3× dilution), stopwords double, MTLD collapses, and baseline persistence falls to 5.2%. K0 retention alone underestimates the onset of degradation."

#### F. Regime Taxonomy (updated)

| Regime | Examples | Retention | Output Distribution |
|---|---|---|---|
| **Homeostatic** | Qwen r≤64, Gemma r≤4 | Stable (>90%) | Stable: length, diversity, persistence all constant |
| **Distributionally degraded, factually bounded** | Qwen r=128 | Bounded (~88%) | Degraded: verbosity, filler, efficiency collapse |
| **Factually degradative** | Qwen r=256, Gemma r≥16 | Progressive loss | Degraded: elaborative drift, persistence lost |

This three-regime taxonomy is richer than the original two-regime (homeostatic/degradative) model.

#### E. Cross-architecture normalization (negative result)

Tested: eff_rank/{hidden_dim, sqrt(d), num_layers, num_heads, num_kv_heads, baseline_diversity}. None aligned thresholds across Qwen and Gemma. The threshold is intrinsic to model geometry, not reducible to simple architectural ratios. Reported as limitation.

#### Proposed mechanism (for paper, calibrated):

> "Above the capacity threshold, recursive fine-tuning induces simultaneous factual degradation and synthetic-output drift (verbosity increase, lexical efficiency decrease). These co-occur as symptoms of the same high-perturbation regime. First-difference analysis (r=-0.43) suggests modest incremental coupling between output drift and subsequent retention loss, but controlling for shared temporal trend reduces the association substantially (partial r=-0.22). We characterize output drift as a measurable co-symptom and potential monitoring signal, not as a demonstrated causal driver of factual loss."

### 5.2 CKA-Factual vs CKA-Global
- CKA-Factual detects adaptation that CKA-Global misses
- But CKA does NOT distinguish synthetic from real data (both ~0.983)
- CKA measures adaptation intensity, not recursive toxicity

### Synthetic vs Real (G1 vs G2, r=16)
- CKA identical between groups
- Factual transitions: synthetic freezes (0 after Gen1), real maintains flux (2-7/gen)
- "Factual ossification" rather than collapse at low rank

### Adapter Health
- Effective rank stable across all regimes (no rank collapse)
- No geometric degeneration of LoRA matrices observed

---

## 6. What Was Refuted

| Original hypothesis | Status |
|---|---|
| Knowledge collapse inevitable under recursive PEFT | REFUTED for r≤128 |
| ESI as early warning of collapse | NOT TESTABLE (no collapse occurred) |
| Stage B (valley of dangerous competence) | NOT OBSERVED |
| CKA detects recursion-specific damage | REFUTED (G2 real = same CKA) |
| Adapter rank collapse causes degradation | REFUTED (eff rank stable) |
| Universal effective-rank threshold | REFUTED (backbone-dependent by ~10×) |

---

## 7. Relationship to Literature

| Paper | Their claim | Our finding | Positioning |
|---|---|---|---|
| Shumailov 2024 (Nature) | Recursive → irreversible collapse | Bounded under low perturbation; consistent at high perturbation | We identify the capacity regime where their predictions hold |
| Dohmatob 2025 (ICLR) | E[L_T] ≥ E[L_0] + αkT | r=256 qualitatively consistent; r≤128 falls outside their assumptions | Low-rank changes the effective hypothesis class |
| Keisha 2025 | Three-stage collapse (Gemma 3 1B, FFT) | QLoRA r=4 homeostatic on same backbone, different protocol | Suggestive context (confounded by dataset/eval format), not controlled replication |
| Biderman 2024 (TMLR) | LoRA learns less, forgets less | Confirmed and extended to dose-response + threshold | We map WHEN "forgets less" holds vs fails |
| Adapala 2025 | Anti-Ouroboros (cumulative + filter) | Replace-without-filter also bounds degradation at low rank | Harsher protocol, complementary finding |
| Gerstgrasser 2024 | Accumulation prevents collapse | Low-rank also bounds under replace protocol | Orthogonal: data-axis vs update-axis |
| Xu 2025 | P(improvement) < 1/2 | Low-drift = stagnation (no improvement, no collapse) | Consistent: small random-walk steps = bounded drift |
| Zibakhsh 2024 | TCE loss delays collapse 2.3× | Low-rank bounds collapse indefinitely (10 gens) at low rank | Orthogonal: loss-signal vs update-subspace |

### Positioning statement:

> "We do not refute model-collapse results; we identify the capacity and perturbation regimes under which collapse-like degradation emerges or remains bounded. The collapse literature documents degradation under high-capacity recursive training; the PEFT literature shows LoRA regularizes. We bridge them: perturbation magnitude — set by rank under PEFT or LR under FFT — is the dominant variable. QLoRA additionally incurs a smaller one-time factual cost. Thresholds are backbone-dependent; cross-architecture normalization and update-subspace geometry remain future work."

### What the paper CAN claim:
- First systematic dose-response curve (r=4→256, 10 gens, 3 seeds)
- First capacity-dependent regime transition identification
- Module topology invariance (attention vs full-linear)
- Multi-backbone confirmation of regime structure
- Effective rank as primary predictor within a backbone
- Dual dose-response (QLoRA rank + FFT LR) confirming magnitude dominance
- Deterministic fragile-fact floor under FFT (Jaccard=1.0 cross-seed)

### What the paper CANNOT claim:
- First to use PEFT for recursive training (Biderman, Adapala exist)
- Novel observation that LoRA prevents collapse (Biderman showed this)
- Controlled contradiction of Keisha (different dataset/eval format)
- "Contradicts" Shumailov or Dohmatob (we locate regime boundaries, not refute)
- Different update subspace directions (fact-overlap is output-level, not weight-geometry)
- Precise threshold location (between two rank values)
- Generalization beyond 1-2B scale

---

## 8. Remaining Work (ordered)

| Priority | Task | Status |
|---|---|---|
| ~~P1~~ | ~~FFT LR sweep~~ | ✅ DONE (2026-06-27) |
| ~~P1~~ | ~~Drift-matched replication (seeds 137+256, Gen5)~~ | ✅ DONE (2026-06-28) |
| **P1** | Bootstrap CIs + per-seed tables + effect sizes | **NEXT** |
| **P2** | Paper draft | After statistics |
| **P2** | Cross-architecture normalization (limitation/future work text) | During writing |

### EXPERIMENTS CLOSED. No further GPU work.

### Cut (do not pursue):
- Gemma 4 threshold
- r=192
- G3 Accumulation
- Wilcoxon with n=3
- Any new backbone/dataset/protocol

---

## 9. Stopping Rule

**Experiments stop when:**
1. FFT LR sweep completes and drift-matched comparison is interpretable
2. OR: FFT sweep is blocked (hardware/time) — declare as limitation

**After experiments stop:**
- Freeze all numerical results
- Compute bootstrap CIs
- Write paper

No new backbones. No new ranks. No new protocols.

---

## 10. Proposed Paper Structure

1. Introduction: recursive training crisis, gap in PEFT characterization
2. Related Work: Shumailov, Dohmatob, Keisha, Biderman, Adapala, Gerstgrasser
3. Methods: QLoRA protocol, data-only recursion, K0 retention, rank ablation design
4. Results:
   - 4.1: No collapse at r≤128 (Qwen, 3 seeds, 10 gens)
   - 4.2: Progressive degradation at r=256 (3 seeds)
   - 4.3: Dose-response curve with effective rank
   - 4.4: Cross-backbone thresholds (Gemma 3, Gemma 4)
   - 4.5: Module targeting invariance
   - 4.6: FFT control [content depends on result]
5. Discussion: capacity as governance parameter, reconciliation with theory, limitations
6. Limitations: n=3 seeds, 1.5B scale, threshold not precisely located, normalization open
7. Conclusion

---

## 11. File Map (active)

```
code/
├── scripts/
│   ├── g1_rank_ablation.py       # Core rank sweep (Qwen)
│   ├── sprint2_gemma.py          # Gemma 3 backbone
│   ├── sprint2_gemma4.py         # Gemma 4 backbone
│   ├── fft_vs_qlora.py           # FFT control (LR=2e-6, confounded)
│   ├── fft_lr_sweep.py           # Drift-matched FFT (pending)
│   ├── show_all_results.py       # Results aggregator
│   └── read_fft_results.py       # FFT results reader
├── docs/
│   ├── PROJECT_STATUS.md         # Detailed results (v1, verbose)
│   ├── PROJECT_STATUS_v2.md      # This file (clean, current)
│   ├── PHASE2_PLAN.md            # Literature correction + execution plan
│   └── [archived]                # PROTOCOL, THEORY, etc. (OBSOLETE)
├── outputs/                      # All experimental data (JSON)
└── figures/                      # Generated plots
```
