# Phase 2: Revised Plan (Post-Literature Review)

**Date:** 2026-06-25  
**Status:** APPROVED — Executing

---

## CRITICAL: Novelty Correction

The claim "no prior work used PEFT/LoRA for recursive training" is **PARTIALLY FALSE**.

### Papers that overlap:

1. **Biderman et al. (TMLR 2024)** — "LoRA Learns Less and Forgets Less"
   - Established that LoRA regularizes, forgets less, maintains diversity vs FFT
   - This is precisely the mechanism we observe
   - DOES NOT do: recursive multi-generation, dose-response, rank sweep

2. **Adapala (2025)** — "Anti-Ouroboros Effect"  
   - Gemma-2b-it + cumulative LoRA adapters + 5 generations of recursive synthetic training
   - Found +6.6% improvement with selection filtering
   - DOES NOT do: rank ablation, capacity threshold mapping, replace-without-filter protocol

3. **Fu et al. (NeurIPS 2025)** — "Self-Verification Provably Prevents Model Collapse"
   - Combines PEFT with recursive synthetic training + verification
   - Different mechanism (verifier-based), not capacity-based

### Our ACTUAL novelty (what survives):

> "LoRA is known to regularize and forget less (Biderman 2024), and has been used in recursive loops (Adapala 2025). We contribute the **systematic mapping of adapter capacity as the control variable** for recursive collapse, identifying a transition between homeostatic and degradative regimes, and demonstrating topological invariance of the protective mechanism across module targeting schemes."

### What the paper should NOT claim:

- ❌ "First to use PEFT/LoRA for recursive training"
- ❌ "Novel observation that LoRA prevents collapse" (Biderman showed this)
- ❌ "LoRA recursive training on Gemma is new" (Adapala did this)

### What the paper CAN claim:

- ✅ First systematic dose-response curve (r=4 → r=256, 10 generations, 3 seeds)
- ✅ First identification of capacity-dependent regime transition
- ✅ First demonstration of topological invariance (attention vs full-linear)
- ✅ Mechanistic refinement: effective rank as predictor, not just "LoRA helps"
- ✅ Reconciliation with Dohmatob/Shumailov (PEFT changes effective hypothesis class)

---

## Execution Sequence (Revised)

### Item 0 — Literature Integration (NO GPU needed)

Update docs/PROJECT_STATUS.md and paper framing:
- Add Biderman, Adapala, Fu to Related Work
- Rewrite contribution as "capacity mapping" not "method novelty"
- Differentiate from Adapala explicitly (we use replace-without-filter, they use cumulative+selection)

### Item 1 — Audit Aggregate Effective Rank (NO GPU needed)

**Key insight from ChatGPT:** `compute_lora_spectrum()` in `g1_rank_ablation.py` runs on the REAL adapter after each generation. The value reported (e.g., "Effective rank: 11.53 / 16") IS the real measurement.

**Task:** Confirm in code that `compute_lora_spectrum()`:
- Is called after training, on the real peft_model of that generation
- Computes mean over ALL active LoRA A/B pairs

If confirmed:
```
aggregate_eff_rank = mean_eff_rank × num_lora_matrices
```
is a valid real measurement, NOT a proxy.

The `extract_per_matrix_rank.py` script = Gen1 sanity check for heterogeneity only.

### Item 2 — Attention r=64 Gen10 (GPU, ~5h)

Run in parallel with Item 1. Only point in transition zone without Gen10.

```bash
uv run python scripts/g1_rank_ablation.py --rank 64 --target attention --seed 15 --generations 10
```

### Item 3 — Replot with Consistent Generations

**Fig 5a:** Retention Gen5 vs Aggregate Effective Rank (all 8 points)  
**Fig 5b:** Retention Gen10 vs Aggregate Effective Rank (only configs with Gen10)

Title: "Aggregate Effective Rank as Primary Predictor of Retention"

Remove: "same curve", "second-order protection", "distributed protection"

### Item 4 — Wording Corrections

- Full r=16: "bounded, quasi-homeostatic; no runaway degradation"
- Aggregate rank: "primary but not exclusive predictor"
- Remove: "protective effect of dispersed constraints"
- Module allocation: "may modulate retention as secondary factor" (underpowered observation)

### Item 5 — Sprint 2: Gemma 3 1B IT

**Purpose:** Confrontation with Keisha (2025), NOT "LoRA recursion on Gemma" (Adapala did that).

**Framing:** "Under the same backbone where Keisha observed three-stage collapse under FFT, QLoRA bounds the degradation trajectory."

**Differentiation from Adapala:**
- We use replace (not cumulative adapters)
- We test dose-response (r=16 vs r=256), not single-rank
- We do NOT use selection filtering
- We measure effective rank and transitions, not just accuracy

**Execution:**
1. `named_modules()` on Gemma 3 1B IT first
2. r=16 seed=15 Gen5
3. r=256 seed=15 Gen5 (or r=128 if VRAM-constrained)
4. Save per-matrix rank in training loop
5. Add timing per generation

---

## Positioning Statements (for paper)

### Gap statement (honest):
> "Existing work explains or mitigates model collapse primarily through data composition (Gerstgrasser 2024), loss design (Zibakhsh 2024), or full-model recursive training dynamics (Shumailov 2024; Dohmatob 2025). While LoRA's regularizing properties are established (Biderman 2024) and recursive LoRA loops have been explored (Adapala 2025), no prior work systematically maps adapter capacity as the control variable for recursive collapse dynamics."

### Contribution statement:
> "We contribute: (1) a five-point dose-response curve demonstrating monotonic capacity-dependent degradation under recursive PEFT; (2) identification of a regime transition between homeostatic (r≤128) and degradative (r≥256) dynamics replicated across seeds; (3) demonstration that module targeting topology does not alter the regime, with aggregate effective rank serving as the primary predictor."

### Positioning vs Dohmatob:
> "PEFT alters the effective hypothesis class and update subspace, so the assumptions of high-capacity recursive retraining do not directly apply. Our r=256 results, showing ~2.4pp/gen linear degradation, are consistent with Dohmatob's predictions when effective capacity approaches the regime where their bounds become operative."

---

## What We Are NOT Doing

| Temptation | Why skip |
|---|---|
| More ranks (r=192) | 5+ points sufficient, novelty is sweep not precision |
| More seeds on Sprint 1 | N=1 is fine for ablation; N=3 is for the core result |
| r=512 | Impossible on hardware + diminishing returns |
| Qwen 7B | Cost/benefit too low |
| Replicate Adapala's cumulative protocol | Not our research question |
