# SESSION STATE — Consolidated (2026-06-29)

## Project: Recursive Synthetic Fine-Tuning — Pressure-Gated Degradation
## Researcher: Julio Azancort (UFPA)
## Status: ALL EXPERIMENTS COMPLETE. Writing phase.
## Target: Engineering Applications of Artificial Intelligence (Elsevier, IF 7.8, A1)

---

## Final Thesis

Recursive synthetic fine-tuning is governed by **effective training pressure** — a composite of update capacity, perturbation magnitude, and synthetic exposure. Degradation emerges when this pressure crosses a backbone-dependent boundary.

The boundary is sharp: ~5% reduction in synthetic exposure shifts r=256 from degradative (65/78) to near-homeostatic (72-73/78), replicated across 3 independent random masks.

---

## Key Results Summary

### 1. QLoRA Rank Dose-Response (Qwen 2.5 1.5B)
| Rank | Gen10 Retention | Eff Rank | Regime | Seeds |
|---|---|---|---|---|
| r=4 | — (96.2% Gen5) | 3.34 | Homeostatic | 1 |
| r=16 | 94.9% ± 0.7% | 11.08 | Homeostatic | 3 |
| r=64 | 91.1% | 30.08 | Homeostatic | 1 |
| r=128 | 88.6% | 50.5 | Bounded (but distributional drift!) | 1 |
| r=256 | 78.0% ± 2.6% | 87.8 | Degradative | 3 |

### 2. Cross-Backbone
- Gemma 3 1B: threshold eff rank ~3-9 (10× lower than Qwen)
- Gemma 4 E2B: homeostatic at eff 5.6 (n=1, threshold not located)

### 3. FFT vs QLoRA (N=3, Gen10)
- QLoRA: 97.4% mean | FFT LR=1e-6: 91.9% mean | Gap: +5.6pp
- Gap is one-time adaptation cost, NOT differential rate (both flat post-Gen1)
- FFT loses same 6 facts deterministically (Jaccard=1.0 cross-seed)
- QLoRA loses different 2-3 facts (Jaccard=0.0 vs FFT)

### 4. FFT LR Sweep (magnitude dose-response)
| LR | Drift | Gen3 Ret |
|---|---|---|
| 1e-6 | 0.39 | 92.3% |
| 5e-6 | 1.56 | 91.0% |
| 1e-5 | 1.95 | 89.7% |
| 2e-5 | 3.48 | 84.6% |

### 5. Distributional Signatures
- Degradative: verbosity drift, content efficiency collapse (0.40→0.13 for r=128)
- Baseline response persistence: drops to 3.8% in r=256 vs stable 36% in r=16
- Two phenotypes: r=128 (filler/stopword) vs r=256 (elaborative/dispersive)
- r=128 case: retention bounded BUT distribution severely degraded (SDI=1.53)
- Three-regime taxonomy: homeostatic / distributionally degraded / factually degradative

### 6. Causal Intervention (Qwen r=256, seed 15)
| Condition | Gen5 | Delta vs C1 |
|---|---|---|
| C1 Normal | 65/78 | — |
| C2 Short-constrained | 71/78 | +7.7pp (MORE tokens!) |
| C3 Length-filtered | 73/78 | +10.3pp (N=3 robust) |
| C4 Canonical | 69/78 | +5.1pp |
| C5 Random downsample | 72-73/78 | +9pp (3 masks, Jaccard=0.015 vs C3) |

Token budget confound ELIMINATED: C3 uses only 5% fewer tokens; C2 uses MORE tokens.
C3 and C5 remove DIFFERENT examples but both stabilize → pressure-sensitivity, not quality-specific.

### 7. Norm Audit
- `compute_lora_drift()` = ||B@A|| = 0.42 (effective delta, comparable to FFT drift 0.39)
- `compute_lora_norm()` = ||A||+||B|| raw = 17.38 (NOT comparable to anything)
- Comparison is "approximately perturbation-matched"

---

## Critical Wording Rules

| SAY | DON'T SAY |
|---|---|
| effective training pressure | rank causes collapse |
| approximately perturbation-matched | exactly matched drift |
| output drift accompanies degradation | output drift causes degradation |
| pressure-gated at a sharp boundary | inevitable collapse above rank X |
| identifies the regime where predictions hold | contradicts Shumailov/Dohmatob |
| suggestive (different protocol) | controlled replication of Keisha |
| fewer and different facts; subspace = future work | different parameter-space directions |
| in the Qwen r=256 boundary case, 3 masks | any 5% universally works |
| diagnostic signature | early warning (not proven temporally) |
| over the observed 10-generation horizon | indefinitely |
| to our knowledge | first ever |

---

## Documents in code/docs/

| File | Purpose | Status |
|---|---|---|
| PROJECT_STATUS_v2.md | Full results + claims | ✅ Final |
| PAPER_OUTLINE.md | Section skeleton with figures/tables | ✅ Final |
| PAPER_STRUCTURE.md | Detailed subsections | ✅ Final |
| TEMPLATE_RULES.md | Elsevier CAS-DC formatting rules | ✅ Final |
| FUTURE_WORK.md | 18 directions for next paper | ✅ Done |
| PHASE2_PLAN.md | Literature + execution (historical) | Archived |
| PROJECT_STATUS.md | Original verbose status (v1) | Archived |

---

## Git Repository
- Remote: https://github.com/Jazancort/llm-knowledge-collapse
- Latest commit: 6e18fee (2026-06-29)
- All results in outputs/ (force-added past .gitignore)

## Hardware
- Local: RTX 3070 8GB (for analysis scripts)
- Athena: RTX 4000 Ada 20GB (for training, SSH configured)
- SSH: `ssh athena` (key auth configured, agent running)

## Next Step
WRITE THE PAPER in Overleaf using cas-dc template.
Template in: `Paradoxo - springer/Overleaf/`
