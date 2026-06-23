# Early Detection of Knowledge Collapse Through Explanation Stability Analysis

Experimental validation of the hypothesis that representational/explanation instability precedes factual degradation in recursively trained LLMs.

## Setup

```bash
cd code
uv sync
```

Requires: NVIDIA GPU with 8GB+ VRAM, CUDA 12.x.

## Running

```bash
uv run python scripts/m1a_validate_infrastructure.py
```

## Milestones

### M1A — Infrastructure Validation (no training)

Validates model loading, inference, hidden state extraction, CKA, attention rollout, ESI, and accuracy measurement.

```bash
python scripts/m1a_validate_infrastructure.py
```

Expected output: all sanity checks pass (CKA self ≈ 1.0, ESI self ≈ 0.0, accuracy > 0).

### M1B — Single Cycle (1 generation)

QLoRA fine-tune → merge → re-evaluate. Validates the recursive pipeline.

```bash
python scripts/m1b_single_cycle.py
```

### M3 — Full Experiment

10 generations × 3 groups × 3 seeds.

```bash
python scripts/run_experiment.py --config configs/training.yaml
```

## Project Structure

```
code/
├── configs/          # YAML configs (model, training, metrics)
├── data/             # Datasets (train_seed, evaluation, probe)
├── src/
│   ├── model/        # Loading, inference, generation
│   ├── evaluation/   # Accuracy, fluency, confidence
│   ├── explainability/ # CKA, attention rollout, ESI
│   ├── finetuning/   # QLoRA, merge (M1B+)
│   └── utils/        # Config, metadata, helpers
├── scripts/          # Executable scripts per milestone
├── outputs/          # Per-generation results
└── logs/
```

## Experimental Design

- **H1 (primary):** Representational instability precedes factual degradation (Lead Time > 0)
- **H2 (secondary):** ESI predicts future accuracy better than perplexity/entropy/TTR
- **Model:** Gemma 3 4B (primary), Qwen2.5-1.5B (pilot)
- **Groups:** G1 Replacement, G2 Real-only, G3 Accumulation
- **Metrics:** ESI, CKA, factual accuracy, confidence, fluency
- **Seeds:** 3 minimum
- **Success criterion:** Lead Time = GC − GFW > 0, consistent across seeds
