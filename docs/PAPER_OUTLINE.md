# PAPER OUTLINE — Capacity-Gated Knowledge Degradation

**Target venue:** AI/ML journal (Springer AIRE, MLST, or similar)
**Working title:** "How Much Rank Is Too Much? Mapping the Capacity Threshold for Knowledge Degradation Under Recursive Synthetic LoRA"
**Status:** Skeleton. FFT section depends on pending experiment.

---

## Title Options

1. "Capacity-Dependent Knowledge Degradation Under Recursive Synthetic Fine-Tuning: A Dose-Response Analysis of Low-Rank Adaptation"
2. "How Much Rank Is Too Much? Mapping the Capacity Threshold for Knowledge Degradation Under Recursive Synthetic LoRA"
3. "Low-Rank Regularization Bounds Recursive Knowledge Degradation: A Multi-Backbone Dose-Response Study"

---

## Abstract (draft skeleton)

Recursive training on synthetic data causes progressive model collapse — but does this hold under parameter-efficient fine-tuning? We systematically map adapter capacity (LoRA rank r=4 to r=256) as the control variable for recursive knowledge degradation across three transformer backbones. We find a capacity-gated regime transition: below a backbone-specific effective-rank threshold, models maintain 88-97% factual retention over 10 recursive generations; above it, retention declines at 2-6 pp/generation. The threshold differs by an order of magnitude across architectures (~50-88 for Qwen 2.5 1.5B vs ~3-9 for Gemma 3 1B), demonstrating that regime structure generalizes but raw thresholds are backbone-dependent. [FFT sentence TBD]. These results position adapter rank as a governance-relevant parameter for recursive training safety.

---

## 1. Introduction (~1.5 pages)

- Recursive training on synthetic/LLM-generated data is increasingly common
- Model collapse (Shumailov 2024) and knowledge collapse (Keisha 2025) documented under FFT
- PEFT/LoRA is the dominant fine-tuning method in practice — but recursive PEFT is understudied
- Biderman (2024) showed LoRA "learns less, forgets less" — but how much rank crosses the line?
- **Gap:** No systematic mapping of adapter capacity as control variable for recursive collapse
- **Contribution:** dose-response curve, regime transition, multi-backbone, module invariance

---

## 2. Related Work (~2 pages)

### 2.1 Model Collapse and Knowledge Collapse
- Shumailov 2024 (Nature): irreversible collapse under replace protocol
- Dohmatob 2025 (ICLR): E[L_T] ≥ E[L_0] + αkT for any k>0
- Seddik 2024: Dirac mass convergence
- Xu 2025: P(improvement) < 1/2, random walk framework
- Keisha 2025: three-stage knowledge collapse, Stage B

### 2.2 Mitigations
- Gerstgrasser 2024: accumulation bounds collapse
- Zibakhsh 2024: TCE loss delays 2.3×
- Yi 2025: verification helps short-term, converges to verifier

### 2.3 PEFT and Recursive Training
- Biderman 2024 (TMLR): LoRA regularizes, forgets less — established the mechanism
- Adapala 2025: Anti-Ouroboros, cumulative LoRA + selection filter, +6.6%
- **Our positioning:** We don't discover that LoRA helps (Biderman did). We map HOW MUCH rank transitions from helpful to harmful.

---

## 3. Methods (~3 pages)

### 3.1 Recursive Protocol
- Data-only recursion: new adapter from scratch each generation, base model frozen
- Replace without filter (no selection, no accumulation)
- TriviaQA: 2000 train, 200 eval, 100 probe
- Greedy decoding for eval; T=0.7 for generation

### 3.2 Models and Hardware
- Qwen 2.5 1.5B-Instruct (primary)
- Gemma 3 1B IT (secondary)
- Gemma 4 E2B IT (robustness)
- RTX 3070 8GB (local) + RTX 4000 Ada 20GB (Athena)

### 3.3 Rank Ablation Design
- r ∈ {4, 16, 32, 64, 128, 256}
- Target modules: q_proj, v_proj (attention-only) AND all linear (full)
- Fixed: LR=1e-5, epochs=2, batch=4×4, BF16

### 3.4 Metrics
- K0 Retention: fraction of Gen0-correct items still correct at Gen T
- Effective rank: exp(H(σ)) of LoRA A·B matrices (mean over all adapter pairs)
- Transition matrices: C→W (forgetting), W→C (recovery)

### 3.5 Multi-Seed Protocol
- Seeds: 15, 137, 256 for headline claims
- Remaining points: seed 15 only (exploratory/ablation)

### 3.6 FFT Control [conditional section]
- Drift-matched comparison: find FFT LR producing same weight drift as QLoRA r=16
- Compare retention at equal perturbation magnitude
- [Content depends on FFT LR sweep result]

---

## 4. Results (~4 pages)

### 4.1 Homeostatic Regime (r≤128, Qwen)
- 94.9% ± 0.7% at Gen10 (r=16, 3 seeds)
- No runaway degradation over 10 generations
- Factual ossification: zero transitions after Gen1

### 4.2 Degradative Regime (r=256, Qwen)
- 78.0% ± 2.6% at Gen10 (3 seeds)
- ~2.4 pp/gen progressive loss
- Plasticity depletion: W→C approaches 0

### 4.3 Dose-Response Curve
- 5 points: monotonic relationship between effective rank and retention loss
- Sublinear effective rank growth with nominal rank
- Diminishing utilization: model recruits proportionally less capacity at higher ranks

### 4.4 Cross-Backbone Generalization
- Gemma 3: threshold at eff rank ~3-9 (10× lower than Qwen)
- Gemma 4: confirms low-rank stability (threshold not located, n=1)
- Regime STRUCTURE generalizes; raw threshold does NOT
- Post-threshold plateau in Gemma (r=16 ≈ r=256)

### 4.5 Module Targeting Invariance
- Attention-only vs Full-Linear: same regime given same effective rank
- Effective rank per adapter is the primary predictor

### 4.6 FFT Control [PENDING]
- [Scenario A: FFT degrades more → PEFT structural protection]
- [Scenario B: FFT matches → drift-mediated stability]
- [Scenario C: FFT stable → protocol-level robustness]

---

## 5. Discussion (~2 pages)

### 5.1 Reconciliation with Theory
- Dohmatob's linear bound applies when effective capacity is high (our r=256 is consistent)
- Low-rank constrains the update subspace → noise cannot accumulate unboundedly
- Not contradiction of Shumailov — complementary mechanism in a different regime

### 5.2 Practical Implications
- Adapter rank as governance parameter
- Simple recommendation: keep effective rank below backbone-specific threshold
- Monitoring effective rank during training as safety check

### 5.3 Why Thresholds Differ Across Architectures
- Hidden dim, depth, GQA head ratio differ
- "Effective rank 9" means different functional perturbation in each architecture
- Normalization remains open (future work)

---

## 6. Limitations (~0.5 page)

- Single dataset (TriviaQA factoid QA)
- Small models (1-2B parameter range)
- n=3 seeds for headline, n=1 for ablation
- Threshold imprecisely located (between two rank values)
- Cross-architecture normalization not solved
- No causal mechanism proposed (observational dose-response)
- [If FFT not completed: FFT comparison as limitation]

---

## 7. Conclusion (~0.5 page)

- Adapter capacity controls recursive collapse outcome
- Below threshold: PEFT provides practical immunity (10 generations, 3 seeds)
- Above threshold: progressive degradation consistent with theoretical predictions
- Adapter rank is a governable, monitorable safety parameter
- [Conditional on FFT: statement about whether protection is PEFT-specific or drift-mediated]

---

## Figures (7 planned)

| Fig | Content | Status |
|---|---|---|
| 1 | Longitudinal retention trajectories (r=16 vs r=128 vs r=256, 10 gens) | Generated |
| 2 | Dose-response: retention and eff rank vs nominal rank | Generated |
| 3 | Plasticity depletion: C→W vs W→C per generation (r=128 vs r=256) | Generated |
| 4 | Utilization curve: eff/nominal rank showing diminishing returns | Generated |
| 5a | Retention vs mean eff rank — Gen5 (all configs) | Generated |
| 5b | Retention vs mean eff rank — Gen10 (configs with Gen10 data) | Generated |
| 6 | FFT vs QLoRA at matched drift | **PENDING** |
| 7 | Cross-backbone threshold comparison (Qwen vs Gemma 3 vs Gemma 4) | To generate |

---

## Statistical Reporting Plan

With n=3 seeds:
- Report all individual seed values (full transparency)
- Bootstrap 95% CIs on retention deltas between regimes
- Effect sizes (Cohen's d analog for proportions)
- Frame as "consistent across seeds with non-overlapping ranges"
- Lean on dose-response monotonicity (5 ranks) as primary statistical strength

Do NOT:
- Wilcoxon with n=3 (underpowered, statistical theater)
- Claim "p << 0.01" (insufficient n)
- Present n=1 results as statistically validated

---

## Key Decisions for Writing

1. Title: decide between mechanistic ("capacity-gated") vs practical ("how much rank")
2. FFT framing: structural protection vs drift-mediated (depends on result)
3. Gemma 4: robustness footnote or supplementary table (not main result)
4. Cross-architecture: present as descriptive finding + open normalization question
5. Positioning: refinement of known mechanism (Biderman), not discovery of PEFT protection
