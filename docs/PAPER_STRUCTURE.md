# Paper Structure — Final (Post-Review)

**Thesis:** Recursive synthetic fine-tuning is governed by effective training pressure — a composite of update capacity, perturbation magnitude, and synthetic exposure. Degradation emerges when this pressure crosses a backbone-dependent boundary.

---

## 1. Introduction

1.1. Recursive synthetic training: the growing problem
1.2. Model collapse vs knowledge collapse
1.3. PEFT/LoRA: dominant method, understudied in recursive context
1.4. Research question: what governs the stability/degradation transition?
1.5. Contributions (effective training pressure framework, regime mapping, intervention)

---

## 2. Related Work

2.1. Model collapse theory (Shumailov, Dohmatob, Seddik, Xu)
2.2. Knowledge collapse and three-stage degradation (Keisha)
2.3. Mitigations: accumulation, filtering, loss shaping (Gerstgrasser, Zibakhsh, Yi)
2.4. PEFT and recursive training (Biderman, Adapala)
2.5. Positioning: we refine prior results by mapping the pressure landscape

---

## 3. Methodology

3.1. Recursive protocol
  - Data-only recursion (new adapter each gen, base frozen)
  - Replace without filter
  - Dataset: TriviaQA (2000 train, 200 eval, K0 subset)

3.2. K0 factual retention metric
  - Definition, normalization, exact match with aliases

3.3. Models and hardware
  - Qwen 2.5 1.5B-Instruct (primary)
  - Gemma 3 1B IT (secondary)
  - Gemma 4 E2B IT (robustness, n=1)

3.4. Experimental axes
  - Axis 1: QLoRA rank (r=4/16/32/64/128/256) — update capacity
  - Axis 2: FFT learning rate (1e-6 to 2e-5) — perturbation magnitude
  - Axis 3: Method comparison (QLoRA vs FFT, drift-matched)
  - Axis 4: Module topology (attention vs full-linear, supportive n=1)
  - Axis 5: Synthetic exposure (C3 filtering, C5 downsampling)

3.5. Effective training pressure (organizing framework, not fitted scalar)

3.6. Output-distribution metrics
  - Content efficiency (retention / mean_length)
  - Baseline response persistence
  - Distinct-n, MTLD, stopword ratio
  - Synthetic Drift Index (SDI-3)

3.7. Intervention design
  - C1 Normal (control)
  - C2 Short-answer constrained
  - C3 Length-filtered
  - C4 Canonical extraction
  - C5 Token-matched random downsampling (3 masks)

3.8. Seeds, statistics, reporting
  - N=3 for headline (seeds 15, 137, 256)
  - N=1 explicitly labeled
  - No Wilcoxon; per-seed values + range

---

## 4. Results

### 4.1. Update Capacity: QLoRA Rank Dose-Response

4.1.1. Five-point monotonic dose-response (Qwen)
4.1.2. Effective rank as within-backbone predictor
4.1.3. Homeostatic stability through Gen10 (r≤128, N=3)
4.1.4. Progressive degradation at r=256 (N=3)
4.1.5. Module-targeting ablation (supportive, n=1)

### 4.2. Cross-Backbone Thresholds

4.2.1. Gemma 3: threshold at eff rank ~3-9 (N=3)
4.2.2. Gemma 4: robustness confirmation (n=1, threshold not located)
4.2.3. Post-threshold plateau in Gemma 3 (r=16 ≈ r=256)
4.2.4. Architectural normalization: attempted and failed (limitation)
4.2.5. Regime structure generalizes; raw threshold does not

### 4.3. Perturbation Magnitude: FFT Learning Rate Sweep

4.3.1. FFT dose-response: drift monotonically predicts retention
4.3.2. FFT vs QLoRA drift-matched (N=3, Gen10)
4.3.3. One-time adaptation cost (~5.6pp), not differential degradation rate
4.3.4. Fragile-fact floor: FFT Jaccard=1.0 (same 6 facts, all seeds)
4.3.5. QLoRA vs FFT: different facts lost (Jaccard=0.0)

### 4.4. Distributional Signatures of Degradation

4.4.1. Homeostatic: stable length, diversity, persistence
4.4.2. Degradative: verbosity drift, content efficiency collapse
4.4.3. Baseline response persistence loss (3.8% at Gen10 in r=256)
4.4.4. Two degradation phenotypes (r=128 filler vs r=256 elaborative)
4.4.5. Three-regime taxonomy
4.4.6. r=128 dissociation: retention bounded, distribution degraded
4.4.7. Output drift is diagnostic signature, not proven causal driver
4.4.8. Cross-lag robustness: accompanies, does not demonstrably precede

### 4.5. Synthetic Exposure: Pressure Sensitivity at the Boundary

4.5.1. Intervention design and rationale
4.5.2. C3 length-filtered: +10pp, robust across N=3 seeds
4.5.3. C2 short-constrained: +7.7pp despite MORE training tokens
4.5.4. Token-budget confound eliminated (54k vs 57k = 5%)
4.5.5. C5 random downsampling: +9pp with DIFFERENT examples (Jaccard=0.015)
4.5.6. C5 replicated across 3 independent masks (all 72-73/78)
4.5.7. Item-level: C3 loses only 1 new fact vs ~8 in baseline
4.5.8. Conclusion: r=256 sits at a sharp boundary; ~5% pressure reduction restores stability

---

## 5. Discussion

5.1. Effective training pressure as unifying framework
  - Three knobs: capacity, magnitude, exposure
  - Sharp boundary, not gradual
  - Multiple control points available

5.2. Why QLoRA helps: pressure control, not prevention
  - Rank constrains capacity (one knob)
  - Does not prevent collapse at high rank
  - One-time cost advantage over FFT

5.3. Why thresholds differ across backbones
  - Hidden dim, depth, architecture-specific factors
  - Not reducible to simple normalization
  - Future work direction

5.4. Output drift as diagnostic signal
  - Marks above-threshold regimes reliably
  - Can appear before factual collapse (r=128)
  - Not the sole causal driver (C5 evidence)
  - Practical monitoring value

5.5. The r=128 case: hidden distributional degradation
  - Retention alone underestimates degradation
  - Implications for evaluation and safety

5.6. Practical governance implications
  - Monitor: effective rank, output length, content efficiency, persistence
  - Act: reduce rank or exposure if signals degrade
  - Rule of thumb: cap effective training pressure conservatively

5.7. Reconciliation with collapse literature
  - We identify regime boundaries, not refute prior work
  - High-pressure regime consistent with Shumailov/Dohmatob
  - Keisha: suggestive context (same backbone, different protocol)

---

## 6. Limitations

6.1. Model scale (1-2B only)
6.2. Single dataset (TriviaQA factoid)
6.3. Cross-architecture normalization unsolved
6.4. Causal decomposition incomplete
6.5. Perturbation matching approximate
6.6. Pressure sensitivity tested at one rank (r=256)
6.7. No weight-level mechanism
6.8. Gen10 maximum depth
6.9. Module topology and Gemma 4 are n=1

---

## 7. Conclusion

7.1. Summary: pressure-gated recursive degradation
7.2. Sharp boundary enables practical governance
7.3. Multiple knobs: rank, drift, exposure
7.4. Output drift as monitorable diagnostic
7.5. Future work: scale, datasets, causal decomposition, normalization

---

## Supplementary / Appendices

A. Full per-seed retention tables (all configs)
B. Effective rank per generation (all configs)
C. Complete diversity metrics tables
D. Cross-lag robustness checks (first-diff, partial, reverse-lag)
E. Token decomposition per intervention condition
F. Response length distributions per generation
G. Fact-overlap item-level details
H. C5 mask overlap analysis
