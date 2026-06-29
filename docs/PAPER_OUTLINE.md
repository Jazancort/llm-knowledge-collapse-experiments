# PAPER OUTLINE — Pressure-Gated Recursive Degradation

**Working title:** "Effective Training Pressure Gates Recursive Knowledge Degradation Under Low-Rank Adaptation"
**Status:** Final outline. All experiments complete. Ready to write.
**Last updated:** 2026-06-29

---

## Title Options

1. "Effective Training Pressure Gates Recursive Knowledge Degradation in LLMs: A Multi-Axis Dose-Response Study"
2. "Sharp Boundaries: How Adapter Capacity, Perturbation Magnitude, and Synthetic Exposure Jointly Gate Recursive Collapse"
3. "Pressure-Gated Stability in Recursive Synthetic Fine-Tuning: Dose-Response, Diagnostic Signatures, and Causal Intervention"

---

## Abstract (skeleton)

Recursive training on synthetic data causes model collapse — but the transition from stability to degradation is sharper and more controllable than previously understood. We show that recursive factual degradation under QLoRA is governed by *effective training pressure* — a composite of adapter capacity, update magnitude, and synthetic exposure. Through systematic dose-response mapping across three backbones (Qwen 2.5 1.5B, Gemma 3 1B, Gemma 4 E2B), FFT comparison, and causal intervention experiments, we find: (1) a capacity-gated regime transition between homeostatic retention (88-97% over 10 generations) and progressive degradation (~2-6pp/gen); (2) the threshold is backbone-dependent and not reducible to a universal effective-rank value; (3) at approximately matched perturbation, QLoRA retains ~5pp more than FFT (one-time cost, not rate difference); (4) above threshold, output drift (verbosity, efficiency collapse) marks the degradative regime but is not its sole causal driver; (5) removing ~5% of training examples — by any criterion — shifts the system back to near-homeostatic stability (confirmed across 3 random masks). These results position recursive degradation as pressure-gated rather than purely capacity-gated, with practical implications for monitoring and governing recursive training pipelines.

---

## 1. Introduction (~1.5 pages)

- Recursive synthetic training is increasingly common (LLM-generated data in training pipelines)
- Model collapse documented under FFT (Shumailov 2024, Dohmatob 2025, Keisha 2025)
- PEFT/LoRA is dominant fine-tuning method — recursive PEFT understudied
- Biderman (2024): LoRA "forgets less" — but how much is too much? When does it break?
- **Gap:** No systematic mapping of what controls the stability/degradation transition
- **Contribution:** To our knowledge, this is the first systematic mapping of effective training pressure as the governing variable for recursive stability. We map the regime transition, characterize its diagnostic signatures, and demonstrate that marginal pressure reduction restores stability

---

## 2. Related Work (~2 pages)

### 2.1 Model Collapse and Knowledge Collapse
- Shumailov 2024 (Nature): irreversible collapse under replace
- Dohmatob 2025 (ICLR): linear error growth for any k>0
- Keisha 2025: three-stage collapse, Stage B (Gemma 3 1B, FFT)
- Seddik 2024: Dirac mass convergence
- Xu 2025: P(improvement) < 1/2

### 2.2 Mitigations
- Gerstgrasser 2024: accumulation bounds collapse (data axis)
- Zibakhsh 2024: TCE loss delays 2.3× (loss-signal axis)
- Yi 2025: verification helps but converges to verifier

### 2.3 PEFT and Recursive Training
- Biderman 2024 (TMLR): LoRA regularizes, forgets less
- Adapala 2025: Anti-Ouroboros (cumulative + filter)
- **Our positioning:** We don't discover LoRA helps (Biderman did). We map the pressure landscape that determines when it fails, and show the boundary is sharp and multi-axial.

---

## 3. Methods (~3 pages)

### 3.1 Recursive Protocol
- Data-only recursion: new adapter from scratch each generation, base model frozen
- Replace without filter (no selection, no accumulation)
- TriviaQA: 2000 train, 200 eval; K0 retention metric

### 3.2 Models
- Qwen 2.5 1.5B-Instruct (primary)
- Gemma 3 1B IT (secondary)
- Gemma 4 E2B IT (robustness, n=1)

### 3.3 Experimental Axes

| Axis | Variable | Range |
|---|---|---|
| Adapter capacity | QLoRA rank | r=4, 16, 32, 64, 128, 256 |
| Update magnitude | FFT learning rate | 1e-6, 5e-6, 1e-5, 2e-5 |
| Method comparison | QLoRA vs FFT | Drift-matched, N=3, Gen10 |
| Module topology | Attention vs Full-Linear | r=4, r=16 |
| Synthetic exposure | Example count | 100%, ~95% (filtered/downsampled) |

### 3.4 Metrics
- K0 Retention: fraction of Gen0-correct items still correct at Gen T
- Effective rank: exp(H(sigma)) of LoRA B@A matrices
- Content efficiency: retention / mean_response_length
- Baseline response persistence: % items with same answer as Gen0
- Synthetic Drift Index (SDI-3): log(length_ratio) + log(d1_ratio) + instability_increase
- Distinct-1/2, MTLD, stopword ratio

### 3.5 Intervention Design
- C1: Normal (control)
- C2: Short-answer constrained prompt
- C3: Length-filtered (remove >5 words)
- C4: Canonical extraction
- C5: Token-matched random downsampling (3 independent masks)

---

## 4. Results (~5 pages)

### 4.1 Capacity-Gated Regime Transition (QLoRA)
- 5-point dose-response: r=4 (96.2%) → r=256 (78.0% ± 2.6% at Gen10)
- Threshold between r=128 (bounded) and r=256 (degradative)
- Module targeting: retention tracks effective rank better than module location (supportive, n=1)
- **Figure 1:** Longitudinal retention trajectories (r=16, r=128, r=256)
- **Figure 2:** Dose-response curve (retention + eff rank vs nominal rank)

### 4.2 Cross-Backbone Thresholds
- Qwen threshold: eff rank ~50-88
- Gemma 3 threshold: eff rank ~3-9 (10× lower)
- Gemma 4: homeostatic at eff rank 5.6 (threshold not located)
- No simple architectural normalization aligns thresholds
- **Figure 3:** Cross-backbone comparison

### 4.3 FFT vs QLoRA at Matched Perturbation
- FFT LR sweep: dose-response confirms magnitude dominance
- Drift-matched (N=3, Gen10): QLoRA 97.4% vs FFT 91.9% (+5.6pp)
- Gap is one-time adaptation cost, not differential rate (both flat Gen1-10)
- FFT and QLoRA evaluated through Gen10 across three seeds; both homeostatic post-Gen1
- FFT: deterministic fragile-fact floor (Jaccard=1.0, same 6 facts across seeds)
- QLoRA: different facts lost (Jaccard=0.0 vs FFT)
- **Figure 4:** FFT vs QLoRA Gen10 trajectories

### 4.4 Distributional Signatures of Degradation
- Homeostatic: stable length, diversity, persistence
- Degradative: verbosity drift, content efficiency collapse, persistence loss
- Two phenotypes: r=128 (filler/repetitive) vs r=256 (elaborative/dispersive)
- r=128 case: retention bounded but distribution severely degraded (SDI=1.53)
- Three-regime taxonomy: homeostatic / distributionally degraded / factually degradative
- **Figure 5:** Content efficiency + persistence + length over generations

### 4.5 Causal Intervention — Pressure Sensitivity
- C3 length-filtered: +10pp (Gen5 73/78 vs baseline 65/78), N=3 seeds
- C2 short-constrained: +7.7pp despite MORE training tokens
- C5 random downsampling: +9pp with DIFFERENT examples removed (Jaccard=0.015 vs C3)
- C5 replicated across 3 independent masks: all give 72-73/78 (in Qwen r=256)
- Token budget comparable (54k vs 57k, 5% difference)
- **Conclusion:** in this boundary case, ~5% exposure reduction by any method restores stability
- **Figure 6:** Intervention comparison (retention trajectories + token budget)

### 4.6 Cross-Lag and Temporal Analysis
- Raw cross-lag r=-0.965 (degradative) vs 0.000 (homeostatic)
- After controlling for shared trend: partial r=-0.22 (weak)
- Reverse-lag equally strong → bidirectional / common process
- Output drift accompanies degradation, not demonstrated as causal antecedent
- SDI marks regimes but does not separate by architecture

---

## 5. Discussion (~2 pages)

### 5.1 Effective Training Pressure as Governing Variable
- Unifies rank (capacity), LR (magnitude), and exposure (examples) under one framework
- The threshold is sharp: ~5% change flips the regime
- Practical implication: multiple knobs available for governance

### 5.2 Reconciliation with Literature
- Shumailov/Dohmatob: high-capacity regimes consistent with their predictions; we locate where their bounds apply
- Keisha: same backbone (Gemma 3 1B) is homeostatic under QLoRA r=4 (suggestive, different protocol)
- Biderman: we extend "forgets less" to a pressure-dependent regime map
- Gerstgrasser: orthogonal axis (data accumulation vs update pressure)

### 5.3 Output Drift: Diagnostic Signature, Not Proven Cause
- Strong diagnostic marker (present only above threshold)
- Can appear before severe factual collapse (r=128 case: retention bounded, drift severe)
- C5 shows removal of ANY examples stabilizes — not specific to long outputs
- Honest position: drift marks distributional degradation; causal decomposition remains open

### 5.4 Practical Implications
- Adapter rank as monitorable governance parameter
- Content efficiency / SDI as monitoring signals
- Simple rule: if output length grows or persistence drops, reduce pressure

---

## 6. Limitations (~0.5 page)

- Single dataset (TriviaQA factoid QA)
- Small models (1-2B range)
- N=3 for headline claims, N=1 for ablation points (module topology, Gemma 4)
- Cross-architecture normalization unsolved
- Causal decomposition (quality vs quantity vs magnitude) not fully isolated
- Gen10 max depth (unknown behavior at Gen100+)
- FFT perturbation-matching is approximate (||B@A|| vs weight drift)
- C5 pressure-sensitivity tested only at one rank (r=256)
- No causal mechanism at weight/representation level (output-level only)

---

## 7. Conclusion (~0.5 page)

- Recursive factual degradation is pressure-gated, not inevitably rank-determined
- The system sits near a sharp boundary that can be crossed or uncrossed by small adjustments
- Low-rank QLoRA remains bounded over the observed 10-generation horizon
- QLoRA rank is one practical control knob; synthetic exposure is another
- Output drift serves as a diagnostic signature for monitoring
- Adapter rank and synthetic-stream management are governable safety parameters for recursive training pipelines

---

## Figures (8 planned)

| # | Content | Data source | Status |
|---|---|---|---|
| 1 | Longitudinal retention (r=16/128/256, 10 gens) | g1_rank* | Generated (update needed) |
| 2 | Dose-response: retention + eff rank vs rank | rank ablation | Generated (update needed) |
| 3 | Cross-backbone thresholds | Qwen/Gemma3/Gemma4 | To generate |
| 4 | FFT vs QLoRA Gen10 (3 seeds) | fft_drift_gen10 | To generate |
| 5 | Distributional signatures (content eff, persistence, length) | diversity_analysis | To generate |
| 6 | Intervention comparison (C1/C2/C3/C5 retention + token budget) | causal_intervention + c5_masks | To generate |
| 7 | Mechanism diagram (effective training pressure framework) | Conceptual | To create |
| 8 | Critical boundary sensitivity (C5 masks, all ~72/78) | c5_masks | To generate |

---

## Tables (5 planned)

| # | Content |
|---|---|
| 1 | QLoRA rank sweep: all configs with retention, eff rank, regime, seeds |
| 2 | FFT LR sweep: drift, retention, comparison |
| 3 | FFT vs QLoRA Gen10: per-seed, per-gen, with gap |
| 4 | Intervention results: C1-C5, token budget, retention, examples removed |
| 5 | Distributional metrics: content efficiency, persistence, SDI by regime |

---

## Statistical Reporting

- N=3 claims: per-seed values + mean + range (non-overlapping bands)
- N=1 points: explicitly labeled as exploratory/ablation
- No Wilcoxon (n=3 insufficient)
- Bootstrap CIs where appropriate
- Effect sizes for headline comparisons
- Dose-response monotonicity (5 ranks) as primary statistical strength

---

## Key Wording Decisions

| Say | Don't say |
|---|---|
| "effective training pressure" | "rank causes collapse" |
| "approximately perturbation-matched" | "exactly matched drift" |
| "output drift accompanies degradation" | "output drift causes degradation" |
| "pressure-gated at a sharp boundary" | "inevitable collapse above rank X" |
| "identifies the regime where predictions hold" | "contradicts Shumailov/Dohmatob" |
| "suggestive (different protocol)" | "controlled replication of Keisha" |
| "fewer and different facts; subspace = future work" | "different parameter-space directions" |
| "marginal exposure reduction restores stability" | "any 5% universally works" |
| "diagnostic signature" | "early warning" (not proven temporally) |
