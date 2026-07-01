"""Master runner: execute all new experiments in sequence.

Run on Athena:
  uv run python scripts/run_all_new.py

Estimated time:
  1. Gemma 3 intermediate ranks (r=10,12,14): ~3 ranks × 5 gens × 5 min = ~75 min
  2. Rank × LR matrix (Qwen, 9 configs): ~9 × 5 gens × 4 min = ~180 min
  3. FFT Gemma 3 (3 LRs): ~3 × 5 gens × 15 min = ~225 min
  4. Gemma 3 extra seeds (4 configs): ~4 × 5 gens × 5 min = ~100 min
  Total: ~10 hours

All scripts save incrementally per generation. If interrupted, re-run this
script and completed configs will be skipped automatically.
"""
import subprocess
import sys
import time

SCRIPTS = [
    ("1/4 — Gemma 3 Intermediate Ranks (r=10,12,14)", "scripts/exp_gemma3_intermediate_ranks.py"),
    ("2/4 — Rank × LR Matrix (Qwen, 3×3)", "scripts/exp_rank_lr_matrix.py"),
    ("3/4 — FFT Gemma 3 (3 LRs)", "scripts/exp_gemma3_fft.py"),
    ("4/4 — Gemma 3 Extra Seeds (r=4,16 × seeds 42,77)", "scripts/exp_gemma3_extra_seeds.py"),
]


def main():
    print("=" * 70)
    print("  NEW EXPERIMENTS — MASTER RUNNER")
    print("=" * 70)
    print(f"  Scripts to run: {len(SCRIPTS)}")
    for i, (desc, _) in enumerate(SCRIPTS, 1):
        print(f"    {desc}")
    print("=" * 70)

    total_start = time.time()

    for desc, script in SCRIPTS:
        print(f"\n{'#' * 70}")
        print(f"# {desc}")
        print(f"# Script: {script}")
        print(f"{'#' * 70}\n")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(__import__("pathlib").Path(__file__).parent.parent),
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n  ⚠️  FAILED (exit code {result.returncode}) after {elapsed/60:.1f} min")
            print(f"  Continuing to next script...")
        else:
            print(f"\n  ✓ Completed in {elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE — Total time: {total_elapsed/3600:.1f} hours")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
