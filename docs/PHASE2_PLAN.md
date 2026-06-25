# Phase 2: Methodological Armoring — Experiment Plan

**Status:** PROPOSED (pending grill)  
**Date:** 2026-06-24  
**Prerequisite:** Core experiments complete (dose-response, multi-seed, 10-gen)

---

## Strategic Goal

Transform the paper from "we observed this on Qwen with q_proj/v_proj" into "we observed a capacity-dependent regime under multiple configurations, architectures, and training methods."

Each sprint addresses a specific reviewer attack vector.

---

## Sprint 1 — Full Linear LoRA (MLP Ablation)

### Reviewer Attack

> "Your homeostasis result is an artifact of restricting LoRA to attention only. The MLP layers are where factual knowledge is stored (Meng et al., 2022). Of course restricting adaptation to attention preserves facts."

### Design

**Model:** Qwen2.5-1.5B-Instruct (same as all prior experiments)

**Target modules:**

| Config | Modules | Approx Params (r=16) |
|---|---|---|
| ATTENTION_ONLY (current) | q_proj, v_proj | 2.2M |
| FULL_LINEAR (new) | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | ~7.7M |

**Runs:**

| ID | Rank | Seed | Generations | Target |
|---|---|---|---|---|
| A1.1 | 16 | 15 | 5 | FULL_LINEAR |
| A1.2 | 256 | 15 | 5 | FULL_LINEAR |

**Metrics:** K0 retention, accuracy, transitions (C→W, W→C), effective rank, trainable params

### Decision Tree

| Outcome | Interpretation | Next action |
|---|---|---|
| r=16 stable (>90%), r=256 degrades | Phenomenon is robust to module choice | Proceed to Sprint 2 |
| r=16 degrades slightly (85-90%), r=256 degrades more | MLP amplifies but doesn't change regime | Report as additional finding, proceed |
| r=16 collapses (<80%) | Protection was partially attention-restriction artifact | STOP. Reframe paper. Add as critical finding. |

### Execution Order

1. Run A1.1 (r=16) first — validates the homeostatic regime survives
2. Only if A1.1 passes (>85% retention at Gen 5): run A1.2 (r=256)

---

## Sprint 2 — Architectural Generalization (Gemma)

### Reviewer Attack

> "These results are specific to Qwen's architecture. Without cross-architecture validation, the claimed 'threshold' may be a quirk of Qwen's attention head configuration."

### Design

**Model:** google/gemma-2-2b-it (or gemma-2b if VRAM constrained)

**Runs:**

| ID | Rank | Seed | Generations | Target |
|---|---|---|---|---|
| B1.1 | 16 | 15 | 3 | ATTENTION_ONLY |
| B1.2 | 256 | 15 | 3 | ATTENTION_ONLY |

**Metrics:** K0 retention, transitions, effective rank (no CKA needed — just behavioral replication)

### Success Criterion

Does NOT need to reproduce exact numbers. Needs to reproduce the PATTERN:

```
low rank  → stable
high rank → more degradation
```

### Decision Tree

| Outcome | Interpretation | Next action |
|---|---|---|
| Pattern replicates | Architecture-independent phenomenon | Proceed to Sprint 3 |
| No clear difference between ranks | Qwen-specific or need different rank thresholds for Gemma | Document as limitation, possibly test higher ranks |
| Both collapse | Something fundamentally different about Gemma's factual encoding | Document, do not overfit interpretation |

---

## Sprint 3 — FFT vs QLoRA Direct Comparison

### Reviewer Attack

> "You claim PEFT provides implicit regularization, but you never compared against Full Fine-Tuning. Without this baseline, your 'protection mechanism' claim is unfounded."

### Design

**Model:** Qwen2.5-0.5B-Instruct (smaller model — FFT must fit in 8GB)

**Runs:**

| ID | Method | Seed | Generations |
|---|---|---|---|
| C1.1 | QLoRA r=16 | 15 | 3 |
| C1.2 | Full Fine-Tuning | 15 | 3 |

**Dataset:** Same TriviaQA subset (2000 train, 200 eval)

**Training:** Same hyperparams where applicable (LR may need adjustment for FFT)

### Expected Result

```
FFT:   100 → ~85 → ~72 → ~60  (degradation)
QLoRA: 100 → ~95 → ~95 → ~94  (homeostasis)
```

Numbers don't matter. The DIVERGENCE matters.

### Scientific Value

**Maximum.** This directly connects to Shumailov (2024) and Dohmatob (2025) without philosophical argument. Same model, same data, same recursion — different update method → different outcome.

### Decision Tree

| Outcome | Interpretation | Next action |
|---|---|---|
| FFT degrades, QLoRA stable | Paper's thesis directly demonstrated | Sprint complete |
| Both degrade similarly | LoRA protection claim fails at this scale | Major reframing needed |
| Neither degrades | Model too small to exhibit phenomenon | Try larger model or more generations |

---

## Sprint 4 (Optional) — Dataset Generalization

### Reviewer Attack

> "TriviaQA is a specific format. Results may not generalize to other factual QA benchmarks."

### Design

**Dataset:** Natural Questions (closed-book) or SQuAD

**Runs:**

| ID | Rank | Seed | Generations |
|---|---|---|---|
| D1.1 | 16 | 15 | 3 |
| D1.2 | 256 | 15 | 3 |

### Priority

LOW. Only if Sprints 1-3 succeed and GPU time remains.

TriviaQA is a well-established factual QA benchmark. Most reviewers won't attack dataset choice as the primary concern.

---

## What We Are NOT Doing

| Experiment | Reason |
|---|---|
| More ranks (r=192, r=512) | 5 points already sufficient for dose-response |
| More seeds | N=3 is defensible at workshop/short paper level |
| More generations (>10) | Already exceeds most literature |
| Qwen 7B / Llama 70B | Cost/benefit too low for current venue target |
| Multiple datasets simultaneously | One validation dataset is enough to deflect |

---

## Resource Estimates

| Sprint | Approx GPU Hours | Priority |
|---|---|---|
| Sprint 1 (Full Linear) | ~8-10h | CRITICAL |
| Sprint 2 (Gemma) | ~6-8h | HIGH |
| Sprint 3 (FFT comparison) | ~4-6h | HIGH |
| Sprint 4 (NQ dataset) | ~4-6h | LOW (optional) |

Total if all run: ~22-30h GPU

---

## Open Questions for Grill

1. Is Gemma-2-2b the right choice, or should we use something lighter (Gemma-2B, Phi-3-mini)?
2. For FFT: Qwen 0.5B or TinyLlama 1.1B? (VRAM constraint is 8GB)
3. Should Sprint 3 use the SAME model (Qwen 1.5B with gradient checkpointing) or is a smaller model acceptable?
4. Do we need CKA for Sprints 2-3 or is retention+transitions enough?
5. Is 3 generations sufficient for Sprints 2-3 or do we need 5?
6. Should FULL_LINEAR use the same LR (1e-5) or scale it down given more params?
7. For FFT comparison: same LR as QLoRA or grid-search a reasonable FFT LR?
