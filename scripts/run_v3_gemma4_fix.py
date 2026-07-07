"""
Gemma 4 E2B fix — reduced batch size to avoid OOM on 20GB GPU.
Uses batch=2, grad_accum=8 (same effective batch=16).

Run on Athena:
  tmux attach -t v3exp
  uv run python scripts/run_v3_gemma4_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import run_v3_experiments as exp

# Override: only Gemma 4 experiments
exp.EXPERIMENTS = [
    {"model": "google/gemma-4-E2B-it", "rank": 4, "seed": 15, "label": "gemma4"},
    {"model": "google/gemma-4-E2B-it", "rank": 16, "seed": 15, "label": "gemma4"},
    {"model": "google/gemma-4-E2B-it", "rank": 64, "seed": 15, "label": "gemma4"},
]

# Monkey-patch TrainingArguments to use smaller batch size
_original_run = exp.run_experiment


def patched_run(exp_config, train_questions, k0_indices, k0_questions, k0_answers):
    """Wrapper that reduces batch size for Gemma 4 to avoid OOM."""
    import transformers
    _orig_init = transformers.TrainingArguments.__init__

    def new_init(self, *args, **kwargs):
        kwargs['per_device_train_batch_size'] = 2
        kwargs['gradient_accumulation_steps'] = 8
        _orig_init(self, *args, **kwargs)

    transformers.TrainingArguments.__init__ = new_init
    try:
        return _original_run(exp_config, train_questions, k0_indices, k0_questions, k0_answers)
    finally:
        transformers.TrainingArguments.__init__ = _orig_init


exp.run_experiment = patched_run

if __name__ == "__main__":
    # Clean up failed results from previous OOM run
    import shutil
    for r in [4, 16, 64]:
        result_dir = Path("outputs") / f"v3_gemma4_rank{r}_seed15"
        result_file = result_dir / "results.json"
        if result_file.exists():
            import json
            data = json.loads(result_file.read_text())
            if not data.get("generations"):
                # Empty results from OOM — remove so it restarts clean
                shutil.rmtree(result_dir)
                print(f"  Cleaned up empty {result_dir.name}")

    exp.main()
