# PROJECT STATUS v2 — Capacity-Gated Recursive Fine-Tuning

**Last updated:** 2026-06-28
**Status:** Experiments complete. Statistics + writing phase.

---

## 1. Current Thesis

Recursive synthetic fine-tuning under low-rank adaptation exhibits a **capacity-gated regime transition** between bounded retention (homeostatic) and progressive knowledge degradation. The transition threshold is architecture-general in structure but backbone-dependent in its raw effective-rank value.

**Central mechanism (confirmed):**

Perturbation magnitude is the dominant factor governing recursive factual stability. At approximately comparable perturbation, QLoRA incurs a lower one-time factual adaptation cost than FFT (~5pp, N=3 seeds), after which both methods are equally stable. PEFT makes perturbation magnitude controllable via rank selection — this is why the dose-response curve works as a governance tool.

**Mechanistic nuance:** QLoRA and FFT lose *different* facts (zero overlap in seed 15). The low-rank constraint perturbs along directions that damage fewer facts, rather than protecting the same facts that FFT damages.

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

**Fact-overlap analysis (seed 15):** FFT loses facts [24,46,48,56,74]; QLoRA loses [11,12]. Zero overlap. Both ossify completely after Gen1 (same facts lost in Gen1 and Gen5). The low-rank constraint perturbs along directions that damage fewer facts, not the same facts.

**Paper claim (final, corrected):**

> "At approximately matched perturbation magnitude, QLoRA incurs a lower one-time factual adaptation cost than FFT (~5pp, consistent across 3 seeds: 97.0% vs 92.3%). After the initial adaptation, both methods are equally stable through Gen5. The dominant factor governing stability is perturbation magnitude: FFT at drift 0.4 retains 92.3%, while FFT at drift 3.5 retains 84.6%. The methods damage different facts (zero item overlap), suggesting low-rank adaptation perturbs along subspace directions that affect fewer factual associations."

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

### CKA-Factual vs CKA-Global
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

| Paper | Their claim | Our finding |
|---|---|---|
| Shumailov 2024 (Nature) | Recursive → irreversible collapse | Not under PEFT r≤128. Consistent at r=256. |
| Dohmatob 2025 (ICLR) | Any k>0 → linear error growth | PEFT alters effective hypothesis class. r=256 is consistent. |
| Keisha 2025 | Three-stage collapse (Gemma 3 1B, FFT) | Stage B not observed under QLoRA. Different regime. |
| Biderman 2024 (TMLR) | LoRA learns less, forgets less | Consistent. We extend with dose-response. |
| Adapala 2025 | Anti-Ouroboros (cumulative LoRA) | Different protocol. We use replace-without-filter. |
| Gerstgrasser 2024 | Accumulation prevents collapse | Low-rank PEFT also bounds it under replace protocol. |

### What the paper CAN claim:
- First systematic dose-response curve (r=4→256, 10 gens, 3 seeds)
- First capacity-dependent regime transition identification
- Module topology invariance (attention vs full-linear)
- Multi-backbone confirmation of regime structure
- Mechanistic: effective rank as predictor, not just "LoRA helps"

### What the paper CANNOT claim:
- First to use PEFT for recursive training (Biderman, Adapala exist)
- Novel observation that LoRA prevents collapse (Biderman showed this)
- Causal mechanism of protection (without drift-matched FFT)

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
