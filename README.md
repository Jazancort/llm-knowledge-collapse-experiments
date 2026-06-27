# Capacity-Dependent Knowledge Degradation Under Recursive Synthetic Fine-Tuning

Systematic mapping of adapter capacity as the control variable for recursive collapse dynamics in LLMs under QLoRA.

## Core Finding

Recursive synthetic fine-tuning under QLoRA exhibits a **capacity-gated regime transition**:
- **Homeostatic** (below backbone-specific effective-rank threshold): retention stable at 88-97% over 10 generations
- **Degradative** (above threshold): progressive loss at ~2-6 pp/generation

Regime structure is architecture-general; raw effective-rank threshold is backbone-dependent.

## Setup

```bash
cd code
uv sync
```

Requires: NVIDIA GPU with 8GB+ VRAM, CUDA 12.x, Python 3.12.

## Backbones Tested

| Backbone | Homeostatic | Degradative | Threshold (eff rank) | Seeds |
|---|---|---|---|---|
| Qwen 2.5 1.5B | r≤128 (94.9%) | r=256 (78.0% ± 2.6%) | ~50–88 | 3 |
| Gemma 3 1B IT | r≤4 (92.2%) | r≥16 (68.8% ± 1.2%) | ~3–9 | 3 |
| Gemma 4 E2B IT | r≤16 (97.4%) | not located | >5.6 | 1 |

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/g1_rank_ablation.py` | Rank ablation (r=4/16/32/64/128/256) |
| `scripts/sprint2_gemma.py` | Gemma 3 backbone experiments |
| `scripts/sprint2_gemma4.py` | Gemma 4 backbone experiments |
| `scripts/fft_vs_qlora.py` | FFT vs QLoRA causal control |
| `scripts/fft_lr_sweep.py` | FFT LR sweep for drift-matching |
| `scripts/show_all_results.py` | Aggregate all results |

## Status

See `docs/PROJECT_STATUS.md` for full experimental narrative and results.

**Open:** FFT vs QLoRA causal control (drift-matched validation pending).
