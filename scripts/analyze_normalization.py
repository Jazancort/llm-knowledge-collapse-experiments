"""Analysis B: Cross-architecture normalization — can we find an invariant?

Tests whether normalizing effective rank by some architectural/data factor
aligns the homeostatic/degradative thresholds across Qwen and Gemma 3.

Zero GPU. Uses results from diversity analysis + known effective ranks.

Run: uv run python scripts/analyze_normalization.py
"""
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "diversity_analysis"

# Known architecture parameters
ARCH = {
    "qwen": {"hidden_dim": 1536, "num_layers": 28, "num_heads": 12, "num_kv_heads": 2, "params": "1.5B"},
    "gemma3": {"hidden_dim": 1152, "num_layers": 26, "num_heads": 4, "num_kv_heads": 1, "params": "1B"},
    "gemma4": {"hidden_dim": 1152, "num_layers": 26, "num_heads": 4, "num_kv_heads": 1, "params": "~2B"},
}

# Known effective ranks and regimes (from experiments)
DATA_POINTS = [
    # (model, rank, eff_rank, retention_gen5, regime)
    ("qwen", "r4", 3.34, 96.2, "homeostatic"),
    ("qwen", "r16", 11.08, 94.9, "homeostatic"),
    ("qwen", "r32", 18.0, 94.9, "homeostatic"),
    ("qwen", "r64", 30.08, 89.9, "homeostatic"),
    ("qwen", "r128", 50.5, 87.3, "bounded"),
    ("qwen", "r256", 87.8, 83.1, "degradative"),
    ("gemma3", "r2", 1.8, 97.9, "homeostatic"),
    ("gemma3", "r4", 3.0, 92.2, "homeostatic"),
    ("gemma3", "r16", 9.3, 68.8, "degradative"),
    ("gemma3", "r256", 62.0, 70.2, "degradative"),
    ("gemma4", "r4", 2.1, 96.1, "homeostatic"),
    ("gemma4", "r16", 5.6, 97.4, "homeostatic"),
]


def main():
    print("=" * 80)
    print("CROSS-ARCHITECTURE NORMALIZATION ANALYSIS")
    print("=" * 80)

    # Load diversity data to get Gen0 baseline diversity per model
    csv_path = RESULTS_DIR / "diversity_all.csv"
    gen0_diversity = {}
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["generation"]) == 0:
                    model = row["model"]
                    if model not in gen0_diversity:
                        gen0_diversity[model] = {
                            "distinct_1": float(row["distinct_1"]),
                            "distinct_2": float(row["distinct_2"]),
                            "token_entropy": float(row["token_entropy"]),
                            "mean_length": float(row["mean_length"]),
                        }

    print("\n  Gen0 baseline diversity per model:")
    for model, div in gen0_diversity.items():
        print(f"    {model}: d1={div['distinct_1']:.3f} d2={div['distinct_2']:.3f} "
              f"entropy={div['token_entropy']:.1f} len={div['mean_length']:.1f}")

    # Test normalization candidates
    print("\n" + "-" * 80)
    print("  CANDIDATE NORMALIZATIONS")
    print("-" * 80)

    candidates = {}

    for model, rank, eff_rank, ret, regime in DATA_POINTS:
        arch = ARCH[model]
        hidden = arch["hidden_dim"]
        layers = arch["num_layers"]
        heads = arch["num_heads"]
        kv_heads = arch["num_kv_heads"]

        norm_hidden = eff_rank / hidden
        norm_sqrt_hidden = eff_rank / (hidden ** 0.5)
        norm_layers = eff_rank / layers
        norm_heads = eff_rank / heads
        norm_kv = eff_rank / kv_heads
        norm_hidden_layers = eff_rank / (hidden * layers) * 1000  # scale for readability

        key = f"{model}_{rank}"
        candidates[key] = {
            "model": model, "rank": rank, "eff_rank": eff_rank,
            "retention": ret, "regime": regime,
            "raw": eff_rank,
            "/ hidden_dim": norm_hidden,
            "/ sqrt(hidden)": norm_sqrt_hidden,
            "/ num_layers": norm_layers,
            "/ num_heads": norm_heads,
            "/ num_kv_heads": norm_kv,
            "/ (H*L)*1000": norm_hidden_layers,
        }

    # Print table
    norm_names = ["raw", "/ hidden_dim", "/ sqrt(hidden)", "/ num_layers", "/ num_heads", "/ num_kv_heads"]
    print(f"\n  {'Config':<18} {'Regime':<13} ", end="")
    for n in norm_names:
        print(f"{n:>14}", end="")
    print()
    print("  " + "-" * 105)

    for key in sorted(candidates.keys()):
        c = candidates[key]
        print(f"  {key:<18} {c['regime']:<13} ", end="")
        for n in norm_names:
            print(f"{c[n]:>14.4f}", end="")
        print()

    # Check which normalization best separates regimes
    print("\n" + "-" * 80)
    print("  SEPARATION ANALYSIS: Does any normalization align thresholds?")
    print("-" * 80)

    for norm_name in norm_names:
        homeo_vals = [c[norm_name] for c in candidates.values() if c["regime"] == "homeostatic"]
        degrad_vals = [c[norm_name] for c in candidates.values() if c["regime"] == "degradative"]
        bounded_vals = [c[norm_name] for c in candidates.values() if c["regime"] == "bounded"]

        if not homeo_vals or not degrad_vals:
            continue

        max_homeo = max(homeo_vals)
        min_degrad = min(degrad_vals)
        gap = min_degrad - max_homeo
        separated = gap > 0

        # How well does it separate?
        # Ideal: all homeostatic below threshold, all degradative above
        print(f"\n  {norm_name}:")
        print(f"    Homeostatic range: [{min(homeo_vals):.4f}, {max_homeo:.4f}]")
        if bounded_vals:
            print(f"    Bounded range:     [{min(bounded_vals):.4f}, {max(bounded_vals):.4f}]")
        print(f"    Degradative range: [{min_degrad:.4f}, {max(degrad_vals):.4f}]")
        print(f"    Gap (min_degrad - max_homeo): {gap:.4f} {'SEPARATED' if separated else 'OVERLAPPING'}")

    # Check if eff_rank / Gen0_diversity aligns better
    print("\n" + "-" * 80)
    print("  CAPACITY / DIVERSITY RATIO")
    print("-" * 80)

    if gen0_diversity:
        print(f"\n  {'Config':<18} {'Regime':<13} {'eff_rank':>10} {'d1_gen0':>8} {'ratio':>10}")
        print("  " + "-" * 60)
        ratio_homeo = []
        ratio_degrad = []
        for model, rank, eff_rank, ret, regime in DATA_POINTS:
            if model in gen0_diversity:
                d1 = gen0_diversity[model]["distinct_1"]
                ratio = eff_rank / d1
                print(f"  {model}_{rank:<12} {regime:<13} {eff_rank:>10.2f} {d1:>8.3f} {ratio:>10.2f}")
                if regime == "homeostatic":
                    ratio_homeo.append(ratio)
                elif regime == "degradative":
                    ratio_degrad.append(ratio)

        if ratio_homeo and ratio_degrad:
            gap = min(ratio_degrad) - max(ratio_homeo)
            print(f"\n    Homeostatic ratio range: [{min(ratio_homeo):.2f}, {max(ratio_homeo):.2f}]")
            print(f"    Degradative ratio range: [{min(ratio_degrad):.2f}, {max(ratio_degrad):.2f}]")
            print(f"    Gap: {gap:.2f} {'SEPARATED' if gap > 0 else 'OVERLAPPING'}")


if __name__ == "__main__":
    main()
