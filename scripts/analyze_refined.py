"""Analysis A+B+C Refined: Robust diversity metrics, cross-lag temporal analysis,
fuzzy item stability, and Synthetic Drift Index.

Addresses:
- distinct-1 sensitivity to length (adds absolute counts, normalized entropy, verbosity ratio)
- Temporal precedence: does output drift precede retention drop?
- Fuzzy item stability (token overlap F1 instead of exact match)
- Synthetic Drift Index as candidate regime separator

Zero GPU. Run: uv run python scripts/analyze_refined.py
"""
import json, re, csv, math
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "diversity_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

# Configs with retention data available
CONFIGS_WITH_RETENTION = [
    # (dir, model, rank, regime, retention_by_gen: {gen: retention/78})
    ("g1_gen10_seed15", "qwen", "r16", "homeostatic",
     {0: 1.0, 1: .949, 2: .949, 3: .949, 4: .949, 5: .949, 6: .949, 7: .949, 8: .949, 9: .949, 10: .949}),
    ("g1_rank256_seed15", "qwen", "r256", "degradative",
     {0: 1.0, 1: .936, 2: .923, 3: .897, 4: .872, 5: .831, 6: .808, 7: .795, 8: .782, 9: .782, 10: .780}),
    ("g1_rank128_seed15", "qwen", "r128", "bounded",
     {0: 1.0, 1: .910, 2: .897, 3: .885, 4: .872, 5: .873, 6: .885, 7: .885, 8: .885, 9: .885, 10: .886}),
    ("g1_rank64_seed15", "qwen", "r64", "homeostatic",
     {0: 1.0, 1: .923, 2: .910, 3: .910, 4: .897, 5: .899, 6: .910, 7: .910, 8: .910, 9: .910, 10: .911}),
    ("gemma3_rank4_seed15", "gemma3", "r4", "homeostatic",
     {0: 1.0, 1: .936, 2: .936, 3: .936, 4: .923, 5: .922, 6: .922, 7: .936, 8: .936, 9: .936, 10: .936}),
    ("gemma3_rank16_seed15", "gemma3", "r16", "degradative",
     {0: 1.0, 1: .846, 2: .756, 3: .718, 4: .692, 5: .688}),
]


def extract_responses(texts):
    responses = []
    for text in texts:
        matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
        if matches:
            responses.append(matches[-1].strip())
        else:
            matches = re.findall(r"<start_of_turn>model\n(.*?)(?:<end_of_turn>|$)", text, re.DOTALL)
            if matches:
                responses.append(matches[-1].strip())
            else:
                responses.append(text.strip().split("\n")[-1].strip())
    return responses


def normalize_text(text):
    """Normalize for fuzzy comparison."""
    text = text.lower().strip()
    text = re.sub(r"\b(the|a|an|is|was|are|were)\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(a, b):
    """Token-level F1 between two strings."""
    tokens_a = set(normalize_text(a).split())
    tokens_b = set(normalize_text(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    if not inter:
        return 0.0
    p = len(inter) / len(tokens_a)
    r = len(inter) / len(tokens_b)
    return 2 * p * r / (p + r)


def compute_robust_metrics(responses):
    """Compute length-robust diversity metrics."""
    if not responses:
        return {}

    all_tokens = []
    for r in responses:
        all_tokens.extend(r.lower().split())

    total_tokens = len(all_tokens)
    if total_tokens == 0:
        return {}

    token_counts = Counter(all_tokens)
    unique_count = len(token_counts)

    # Standard metrics
    distinct_1 = unique_count / total_tokens
    bigrams = set(zip(all_tokens[:-1], all_tokens[1:]))
    distinct_2 = len(bigrams) / max(total_tokens - 1, 1)

    # Length-robust: absolute unique count
    unique_unigrams = unique_count
    unique_bigrams = len(bigrams)

    # Normalized entropy (0-1 scale)
    entropy = -sum((c / total_tokens) * math.log2(c / total_tokens) for c in token_counts.values())
    max_entropy = math.log2(unique_count) if unique_count > 1 else 1
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Response stats
    lengths = [len(r.split()) for r in responses]
    mean_length = sum(lengths) / len(lengths)

    # Uniqueness
    unique_responses = len(set(r.lower().strip() for r in responses))
    response_uniqueness = unique_responses / len(responses)

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "unique_unigrams": unique_unigrams,
        "unique_bigrams": unique_bigrams,
        "norm_entropy": norm_entropy,
        "mean_length": mean_length,
        "response_uniqueness": response_uniqueness,
        "total_tokens": total_tokens,
    }


def compute_fuzzy_stability(responses_a, responses_b):
    """Compute mean token-F1 between paired responses across generations."""
    n = min(len(responses_a), len(responses_b))
    if n == 0:
        return 0.0, 0.0
    f1_scores = [token_f1(responses_a[i], responses_b[i]) for i in range(n)]
    exact_matches = sum(1 for i in range(n) if responses_a[i].lower().strip() == responses_b[i].lower().strip())
    return sum(f1_scores) / n, exact_matches / n


def main():
    print("=" * 80)
    print("REFINED ANALYSIS: Robust Metrics + Cross-Lag + Fuzzy Stability + Drift Index")
    print("=" * 80)

    all_results = []

    for dir_name, model, rank, regime, retention_map in CONFIGS_WITH_RETENTION:
        config_dir = OUTPUT_DIR / dir_name
        if not config_dir.exists():
            continue

        print(f"\n{'='*70}")
        print(f"  {dir_name} ({regime})")
        print(f"{'='*70}")

        # Load all generations
        gen_data = {}
        for gen in range(0, 11):
            syn_file = config_dir / f"synthetic_gen{gen}.json"
            if syn_file.exists():
                raw = json.load(open(syn_file))
                gen_data[gen] = extract_responses(raw)

        if not gen_data:
            continue

        gens = sorted(gen_data.keys())
        gen0_metrics = compute_robust_metrics(gen_data[gens[0]])

        print(f"\n  {'Gen':<5} {'d1':>6} {'d2':>6} {'uniq_w':>7} {'norm_ent':>8} "
              f"{'len':>5} {'resp_uniq':>9} {'fuzzy_stab':>10} {'retention':>9}")
        print(f"  {'-'*75}")

        for i, gen in enumerate(gens):
            metrics = compute_robust_metrics(gen_data[gen])

            # Fuzzy stability vs previous gen
            fuzzy_stab = "-"
            if i > 0:
                prev_gen = gens[i - 1]
                f1_mean, exact_pct = compute_fuzzy_stability(gen_data[prev_gen], gen_data[gen])
                fuzzy_stab = f"{f1_mean:.3f}"

            ret = retention_map.get(gen, None)
            ret_str = f"{ret:.3f}" if ret else "-"

            print(f"  {gen:<5} {metrics['distinct_1']:>6.3f} {metrics['distinct_2']:>6.3f} "
                  f"{metrics['unique_unigrams']:>7} {metrics['norm_entropy']:>8.4f} "
                  f"{metrics['mean_length']:>5.1f} {metrics['response_uniqueness']:>9.3f} "
                  f"{fuzzy_stab:>10} {ret_str:>9}")

            all_results.append({
                "config": dir_name, "model": model, "rank": rank, "regime": regime,
                "gen": gen, "retention": ret,
                **metrics,
                "verbosity_ratio": metrics["mean_length"] / gen0_metrics["mean_length"] if gen0_metrics["mean_length"] > 0 else 1,
                "distinct1_ratio": metrics["distinct_1"] / gen0_metrics["distinct_1"] if gen0_metrics["distinct_1"] > 0 else 1,
            })

        # Synthetic Drift Index (Gen0 vs GenFinal)
        final_metrics = compute_robust_metrics(gen_data[gens[-1]])
        length_ratio = final_metrics["mean_length"] / gen0_metrics["mean_length"]
        d1_ratio = gen0_metrics["distinct_1"] / max(final_metrics["distinct_1"], 0.001)
        drift_index = math.log(length_ratio) + math.log(d1_ratio)
        print(f"\n  Synthetic Drift Index (Gen0 vs Gen{gens[-1]}): {drift_index:.3f}")
        print(f"    length_ratio={length_ratio:.2f}, d1_drop_ratio={d1_ratio:.2f}")

    # Cross-lag summary
    print("\n" + "=" * 80)
    print("CROSS-LAG: Does diversity at Gen t predict retention at Gen t+1?")
    print("=" * 80)

    for dir_name, model, rank, regime, retention_map in CONFIGS_WITH_RETENTION:
        config_dir = OUTPUT_DIR / dir_name
        if not config_dir.exists():
            continue

        gen_data = {}
        for gen in range(0, 11):
            syn_file = config_dir / f"synthetic_gen{gen}.json"
            if syn_file.exists():
                gen_data[gen] = extract_responses(json.load(open(syn_file)))

        gens = sorted(gen_data.keys())
        if len(gens) < 3:
            continue

        # Compute diversity at t, retention at t+1
        pairs = []
        for i in range(len(gens) - 1):
            g = gens[i]
            g_next = gens[i + 1]
            if g in retention_map and g_next in retention_map:
                m = compute_robust_metrics(gen_data[g])
                pairs.append((m["mean_length"], m["distinct_1"], retention_map[g_next]))

        if len(pairs) >= 3:
            lengths = [p[0] for p in pairs]
            d1s = [p[1] for p in pairs]
            rets = [p[2] for p in pairs]

            # Simple correlation (Pearson)
            def pearson(x, y):
                n = len(x)
                mx, my = sum(x) / n, sum(y) / n
                num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
                den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
                return num / den if den > 0 else 0

            r_len_ret = pearson(lengths, rets)
            r_d1_ret = pearson(d1s, rets)
            print(f"\n  {dir_name} ({regime}):")
            print(f"    corr(length_t, retention_t+1) = {r_len_ret:+.3f}")
            print(f"    corr(distinct1_t, retention_t+1) = {r_d1_ret:+.3f}")

    # Drift Index comparison
    print("\n" + "=" * 80)
    print("SYNTHETIC DRIFT INDEX — REGIME SEPARATION")
    print("=" * 80)

    drift_indices = {}
    for row in all_results:
        cfg = row["config"]
        if cfg not in drift_indices:
            drift_indices[cfg] = {"regime": row["regime"], "model": row["model"], "rank": row["rank"]}
        if row["gen"] == 0:
            drift_indices[cfg]["d1_gen0"] = row["distinct_1"]
            drift_indices[cfg]["len_gen0"] = row["mean_length"]
        drift_indices[cfg]["d1_final"] = row["distinct_1"]
        drift_indices[cfg]["len_final"] = row["mean_length"]
        drift_indices[cfg]["gen_final"] = row["gen"]

    print(f"\n  {'Config':<30} {'Regime':<13} {'SDI':>6} {'len_ratio':>10} {'d1_ratio':>9}")
    print(f"  {'-'*70}")
    for cfg, d in sorted(drift_indices.items(), key=lambda x: x[1].get("regime", "")):
        if "d1_gen0" in d and "d1_final" in d:
            lr = d["len_final"] / d["len_gen0"]
            dr = d["d1_gen0"] / max(d["d1_final"], 0.001)
            sdi = math.log(lr) + math.log(dr)
            print(f"  {cfg:<30} {d['regime']:<13} {sdi:>6.3f} {lr:>10.2f} {dr:>9.2f}")

    # Save CSV
    csv_path = RESULTS_DIR / "refined_analysis.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
        print(f"\n  Saved: {csv_path}")


if __name__ == "__main__":
    main()
