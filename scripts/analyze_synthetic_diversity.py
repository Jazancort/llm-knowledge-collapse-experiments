"""Analysis A: Synthetic Data Complexity/Diversity across generations and regimes.

Hypothesis: Degradative regimes produce less diverse synthetic data, creating a
positive feedback loop. Homeostatic regimes maintain diversity.

Zero GPU. Runs locally on saved synthetic_genK.json files.

Run: uv run python scripts/analyze_synthetic_diversity.py
"""
import json, re, sys
from pathlib import Path
from collections import Counter
import math
import csv

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "diversity_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

# Configs to analyze: (dir_pattern, model, method, rank_or_lr, seed, regime, max_gen)
CONFIGS = [
    # Qwen QLoRA — dose response
    ("g1_gen10_seed15", "qwen", "qlora", "r16", 15, "homeostatic", 10),
    ("g1_rank4_seed15", "qwen", "qlora", "r4", 15, "homeostatic", 5),
    ("g1_rank32_seed15", "qwen", "qlora", "r32", 15, "homeostatic", 5),
    ("g1_rank64_seed15", "qwen", "qlora", "r64", 15, "homeostatic", 10),
    ("g1_rank128_seed15", "qwen", "qlora", "r128", 15, "bounded", 10),
    ("g1_rank256_seed15", "qwen", "qlora", "r256", 15, "degradative", 10),
    ("g1_rank256_seed137", "qwen", "qlora", "r256", 137, "degradative", 10),
    ("g1_rank256_seed256", "qwen", "qlora", "r256", 256, "degradative", 10),
    # Gemma 3
    ("gemma3_rank4_seed15", "gemma3", "qlora", "r4", 15, "homeostatic", 10),
    ("gemma3_rank4_seed137", "gemma3", "qlora", "r4", 137, "homeostatic", 5),
    ("gemma3_rank4_seed256", "gemma3", "qlora", "r4", 256, "homeostatic", 5),
    ("gemma3_rank16_seed15", "gemma3", "qlora", "r16", 15, "degradative", 5),
    ("gemma3_rank16_seed137", "gemma3", "qlora", "r16", 137, "degradative", 5),
    ("gemma3_rank16_seed256", "gemma3", "qlora", "r16", 256, "degradative", 5),
    ("gemma3_rank256_seed15", "gemma3", "qlora", "r256", 15, "degradative", 5),
    # FFT comparison
    ("fft_comparison_fft_seed15", "qwen", "fft", "lr2e-6", 15, "homeostatic", 5),
    ("fft_comparison_qlora_seed15", "qwen", "qlora", "r16", 15, "homeostatic", 5),
]


def extract_responses(texts):
    """Extract assistant responses from chat-template formatted synthetic data."""
    responses = []
    for text in texts:
        # Find assistant response
        matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
        if matches:
            responses.append(matches[-1].strip())
        else:
            # Try Gemma format
            matches = re.findall(r"<start_of_turn>model\n(.*?)(?:<end_of_turn>|$)", text, re.DOTALL)
            if matches:
                responses.append(matches[-1].strip())
            else:
                # Fallback: take last line
                responses.append(text.strip().split("\n")[-1].strip())
    return responses


def compute_metrics(responses):
    """Compute diversity metrics for a list of response strings."""
    if not responses:
        return {}

    # Tokenize (simple whitespace + lowercase)
    all_tokens = []
    for r in responses:
        all_tokens.extend(r.lower().split())

    total_tokens = len(all_tokens)
    if total_tokens == 0:
        return {}

    # Distinct-n
    unigrams = set(all_tokens)
    bigrams = set(zip(all_tokens[:-1], all_tokens[1:]))
    trigrams = set(zip(all_tokens[:-2], all_tokens[1:-1], all_tokens[2:]))

    distinct_1 = len(unigrams) / total_tokens
    distinct_2 = len(bigrams) / max(total_tokens - 1, 1)
    distinct_3 = len(trigrams) / max(total_tokens - 2, 1)

    # Type-Token Ratio
    ttr = len(unigrams) / total_tokens

    # Token entropy
    token_counts = Counter(all_tokens)
    total = sum(token_counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in token_counts.values() if c > 0)

    # Response uniqueness
    unique_responses = len(set(responses))
    response_uniqueness = unique_responses / len(responses)

    # Duplicate rate
    response_counts = Counter(responses)
    duplicates = sum(1 for c in response_counts.values() if c > 1)
    duplicate_rate = duplicates / len(response_counts) if response_counts else 0

    # Max response frequency (most common answer / total)
    most_common_count = response_counts.most_common(1)[0][1] if response_counts else 0
    max_response_freq = most_common_count / len(responses)

    # Response lengths
    lengths = [len(r.split()) for r in responses]
    mean_length = sum(lengths) / len(lengths)
    std_length = (sum((l - mean_length) ** 2 for l in lengths) / len(lengths)) ** 0.5

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "distinct_3": round(distinct_3, 4),
        "ttr": round(ttr, 4),
        "token_entropy": round(entropy, 4),
        "response_uniqueness": round(response_uniqueness, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "max_response_freq": round(max_response_freq, 4),
        "unique_responses": unique_responses,
        "total_responses": len(responses),
        "mean_length": round(mean_length, 2),
        "std_length": round(std_length, 2),
        "total_tokens": total_tokens,
    }


def main():
    all_rows = []

    for dir_name, model, method, rank_lr, seed, regime, max_gen in CONFIGS:
        config_dir = OUTPUT_DIR / dir_name
        if not config_dir.exists():
            print(f"  SKIP: {dir_name} not found")
            continue

        print(f"\n  Processing: {dir_name} ({regime})")

        for gen in range(0, max_gen + 1):
            syn_file = config_dir / f"synthetic_gen{gen}.json"
            if not syn_file.exists():
                continue

            data = json.load(open(syn_file))
            responses = extract_responses(data)
            metrics = compute_metrics(responses)

            if not metrics:
                continue

            row = {
                "model": model,
                "method": method,
                "rank_or_lr": rank_lr,
                "seed": seed,
                "generation": gen,
                "regime": regime,
                "config": dir_name,
                **metrics,
            }
            all_rows.append(row)
            print(f"    Gen{gen}: d1={metrics['distinct_1']:.3f} d2={metrics['distinct_2']:.3f} "
                  f"uniq={metrics['response_uniqueness']:.3f} entropy={metrics['token_entropy']:.1f} "
                  f"len={metrics['mean_length']:.1f}")

    # Save full results
    csv_path = RESULTS_DIR / "diversity_all.csv"
    if all_rows:
        keys = all_rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Saved: {csv_path} ({len(all_rows)} rows)")

    # Print summary comparison
    print("\n" + "=" * 80)
    print("DIVERSITY ANALYSIS SUMMARY")
    print("=" * 80)

    # Group by regime and show Gen0 vs GenFinal
    for regime_label in ["homeostatic", "bounded", "degradative"]:
        regime_rows = [r for r in all_rows if r["regime"] == regime_label]
        if not regime_rows:
            continue
        print(f"\n  --- {regime_label.upper()} ---")

        # Group by config
        configs = sorted(set(r["config"] for r in regime_rows))
        for cfg in configs:
            cfg_rows = sorted([r for r in regime_rows if r["config"] == cfg], key=lambda x: x["generation"])
            if len(cfg_rows) < 2:
                continue
            g0 = cfg_rows[0]
            gF = cfg_rows[-1]
            d2_delta = gF["distinct_2"] - g0["distinct_2"]
            uniq_delta = gF["response_uniqueness"] - g0["response_uniqueness"]
            ent_delta = gF["token_entropy"] - g0["token_entropy"]
            print(f"    {cfg:40s} Gen0→Gen{gF['generation']}: "
                  f"Δd2={d2_delta:+.4f} Δuniq={uniq_delta:+.4f} Δentropy={ent_delta:+.2f}")


if __name__ == "__main__":
    main()
