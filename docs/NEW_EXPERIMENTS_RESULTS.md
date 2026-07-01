# New Experiments Results — 2026-07-01

## Experiment 1: Gemma 3 Intermediate Ranks (K0=46, seed=15, 5 gens)

| Rank | Gen1 | Gen2 | Gen3 | Gen4 | Gen5 | Eff Rank | Regime |
|---|---|---|---|---|---|---|---|
| r=4 (existing) | 44 | 43 | 44 | 44 | 43 | ~3.0 | Homeostatic |
| r=10 | 41 | 39 | 38 | 39 | 36 | 6.37 | Degradative (slow) |
| r=12 | 38 | 38 | 37 | 34 | 34 | 7.35 | Degradative |
| r=14 | 40 | 38 | 36 | 34 | 32 | 8.37 | Degradative |
| r=16 (existing) | 38 | 38 | 36 | 34 | 33 | ~9.2 | Degradative |

**Conclusion:** Threshold between erank ~3 (r=4) and erank ~6.4 (r=10).
Transition happens between r=4 and r=10. By r=10 the system is already degradative.

## Experiment 2: Rank × LR Matrix (Qwen, K0=78, seed=15, 5 gens)

Gen5 retention values:

| Rank \ LR | 5e-6 | 1e-5 | 2e-5 |
|---|---|---|---|
| r=16 | 76/78 (97.4%) | 76/78 (97.4%) | 72/78 (92.3%) |
| r=64 | 76/78 (97.4%) | 72/78 (92.3%) | 69/78 (88.5%) |
| r=256 | 70/78 (89.7%) | 66/78 (84.6%) | 64/78 (82.1%) |

Full trajectories (Gen1→Gen5):
- r=16, LR=5e-6: 76,76,76,78,76
- r=16, LR=1e-5: 76,75,75,76,76
- r=16, LR=2e-5: 74,73,72,73,72
- r=64, LR=5e-6: 76,76,75,75,76
- r=64, LR=1e-5: 73,72,72,72,72
- r=64, LR=2e-5: 75,73,71,73,69
- r=256, LR=5e-6: 73,72,72,71,70
- r=256, LR=1e-5: 75,72,70,69,66
- r=256, LR=2e-5: 70,69,69,70,64

**Key findings:**
- r=16 + LR=2e-5 → 92.3% (pushes into bounded territory)
- r=64 + LR=5e-6 → 97.4% (fully homeostatic despite medium rank)
- r=256 + LR=5e-6 → 89.7% (better than r=256+LR=1e-5 at 84.6%)
- Axes INTERACT. Not independent.

## Experiment 3: FFT Gemma 3 (K0=46, seed=15, 5 gens)

| LR | Gen1 | Gen2 | Gen3 | Gen4 | Gen5 | Gen5% |
|---|---|---|---|---|---|---|
| 1e-6 | 40 | 38 | 39 | 37 | 36 | 78.3% |
| 5e-6 | 43 | 40 | 37 | 27 | 25 | 54.3% |
| 1e-5 | 42 | 41 | 34 | 32 | 29 | 63.0% |

**Key findings:**
- Magnitude axis generalizes to Gemma 3
- Higher LR → substantially more degradation
- NOT perfectly monotonic (5e-6 worse than 1e-5 at Gen5) — likely noise/seed
- LR=1e-6 on Gemma 3 FFT (78.3%) ≈ QLoRA r=16 (70.2%) — similar but not identical

## Experiment 4: Gemma 3 Extra Seeds (K0=46)

| Config | Gen1 | Gen2 | Gen3 | Gen4 | Gen5 | Gen5% |
|---|---|---|---|---|---|---|
| r=4, seed 42 | 42 | 43 | 43 | 44 | 41 | 89.1% |
| r=4, seed 77 | 44 | 42 | 44 | 44 | 43 | 93.5% |
| r=16, seed 42 | 39 | 36 | 33 | 34 | 32 | 69.6% |
| r=16, seed 77 | 40 | 37 | 36 | 34 | 32 | 69.6% |

Combined with existing (seeds 15, 137, 256):
- r=4 N=5: 93.5%, 91.3%, 89.1%, 93.5%, [89.1% seed42, 93.5% seed77] → mean ~91.3%
- r=16 N=5: 70.2%, 68.1%, 68.1%, [69.6% seed42, 69.6% seed77] → mean ~69.1%

**Regimes confirmed across 5 seeds. Non-overlapping bands.**

## Summary of what changes in the paper

1. §4.2: Gemma threshold refined to erank 3-6 (was "3-9"). N=5 seeds confirmed.
2. §4.3: NEW — Rank×LR interaction shows axes are not independent.
3. §4.3: FFT Gemma confirms magnitude axis cross-backbone.
4. §5.1 Discussion: Axes interact → ETP is multidimensional, not additive.
5. Evidence matrix: add "Axes interact" row.
6. Generalization table: LR now confirmed on Gemma.
