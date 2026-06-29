# Paper Structure — Seções e Subseções

---

## 1. Introduction

1.1. The recursive training problem (synthetic data contamination, scaling)
1.2. Model collapse vs knowledge collapse (distinction)
1.3. The PEFT gap (dominant method, understudied in recursive context)
1.4. Research question and contributions

---

## 2. Related Work

2.1. Model collapse theory (Shumailov, Dohmatob, Seddik, Xu)
2.2. Knowledge collapse and three-stage degradation (Keisha)
2.3. Mitigation strategies (Gerstgrasser accumulation, Zibakhsh TCE, Yi verification)
2.4. PEFT in recursive training (Biderman, Adapala)
2.5. Positioning: what we extend and how we differ

---

## 3. Methodology

3.1. Recursive protocol
  - Data-only recursion (new adapter each gen, base frozen)
  - Replace without filter
  - Dataset: TriviaQA (2000 train, 200 eval, K0 subset)

3.2. Models and hardware
  - Qwen 2.5 1.5B-Instruct (primary)
  - Gemma 3 1B IT (secondary)
  - Gemma 4 E2B IT (robustness)
  - Hardware: RTX 3070 8GB + RTX 4000 Ada 20GB

3.3. Experimental design
  - Axis 1: QLoRA rank ablation (r=4/16/32/64/128/256)
  - Axis 2: FFT learning rate sweep (1e-6 to 2e-5)
  - Axis 3: Method comparison (QLoRA vs FFT, drift-matched)
  - Axis 4: Module topology (attention vs full-linear)
  - Axis 5: Synthetic exposure (filtering, downsampling)

3.4. Metrics
  - K0 factual retention
  - Effective rank (exp entropy of singular values)
  - Content efficiency (retention / mean_length)
  - Baseline response persistence
  - Distinct-n, MTLD, stopword ratio
  - Synthetic Drift Index (SDI-3)

3.5. Multi-seed protocol
  - N=3 for headline claims (seeds 15, 137, 256)
  - N=1 for ablation (explicitly labeled)

---

## 4. Results

### 4.1. Capacity-Gated Regime Transition

4.1.1. QLoRA dose-response (5 ranks, Qwen)
4.1.2. Monotonic relationship: effective rank predicts retention
4.1.3. Homeostatic stability over 10 generations (r≤128)
4.1.4. Progressive degradation at r=256 (3 seeds)
4.1.5. Module targeting invariance

### 4.2. Cross-Backbone Generalization

4.2.1. Gemma 3: lower threshold (eff rank ~3-9)
4.2.2. Gemma 4: robustness at low rank (n=1)
4.2.3. Post-threshold plateau (Gemma 3: r=16 ≈ r=256)
4.2.4. Normalization attempts and failure (limitation)

### 4.3. FFT vs QLoRA: Perturbation Magnitude Dominance

4.3.1. FFT LR sweep: drift dose-response
4.3.2. Drift-matched comparison (N=3, Gen10)
4.3.3. One-time adaptation cost (~5pp), not differential rate
4.3.4. Fragile-fact floor: FFT Jaccard=1.0, QLoRA Jaccard=0.0

### 4.4. Distributional Signatures of Degradation

4.4.1. Output drift: verbosity increase, distinct-1 decrease
4.4.2. Content efficiency collapse (retention / length)
4.4.3. Baseline response persistence loss
4.4.4. Two degradation phenotypes (filler vs elaborative)
4.4.5. Three-regime taxonomy (homeostatic / distributionally degraded / factually degradative)
4.4.6. r=128 dissociation: retention bounded, distribution degraded

### 4.5. Intervention: Pressure Sensitivity at the Boundary

4.5.1. Intervention design (C1-C5)
4.5.2. Length filtering (C3): +10pp, N=3 seeds
4.5.3. Short-constrained (C2): +7.7pp despite more tokens
4.5.4. Token-budget confound eliminated
4.5.5. Random downsampling (C5): +9pp, 3 independent masks
4.5.6. C3 vs C5 overlap: Jaccard=0.015 (different examples, same result)
4.5.7. Critical boundary: ~5% reduction shifts regime

### 4.6. Temporal Analysis (Supplementary)

4.6.1. Cross-lag raw correlations
4.6.2. Robustness checks (first-diff, partial, reverse-lag)
4.6.3. Conclusion: co-symptom, not demonstrated cause

---

## 5. Discussion

5.1. Effective training pressure as unifying framework
  - Capacity + magnitude + exposure = pressure
  - Sharp boundary, not gradual degradation
  - Multiple knobs available for control

5.2. Reconciliation with collapse literature
  - We locate regime boundaries, not refute prior work
  - Shumailov/Dohmatob apply at high pressure
  - Keisha context (same backbone, different protocol)

5.3. Output drift: diagnostic signature, not proven cause
  - Marks above-threshold regimes reliably
  - C5 shows mitigation is not quality-specific
  - Causal decomposition remains open

5.4. Practical implications
  - Rank as governance parameter
  - Content efficiency / SDI as monitoring signals
  - Simple pipeline rule: cap exposure, monitor drift

5.5. The r=128 case: hidden distributional degradation
  - Retention masks underlying output-quality collapse
  - Implications for evaluation beyond accuracy

---

## 6. Limitations

6.1. Scale (1-2B models only)
6.2. Single dataset (TriviaQA factoid)
6.3. Cross-architecture normalization unsolved
6.4. Causal decomposition incomplete (quality vs quantity vs magnitude)
6.5. Perturbation matching approximate (||B@A|| vs FFT drift)
6.6. Pressure sensitivity tested at one rank only (r=256)
6.7. No weight-level mechanism (output-level analysis only)
6.8. Gen10 max depth

---

## 7. Conclusion

7.1. Summary of findings
7.2. Practical recommendations for recursive training governance
7.3. The boundary is sharp — small changes matter

---

## Appendices / Supplementary

A. Full per-seed retention tables
B. Effective rank and LoRA norm per generation
C. Complete diversity metrics (CSV)
D. Cross-lag robustness details
E. Token decomposition per condition
F. Response length distributions
G. Fact-overlap item-level details
