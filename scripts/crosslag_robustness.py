"""Cross-lag robustness checks: first-difference, partial correlation, reverse-lag.

Addresses the criticism that cross-lag r=-0.965 could be inflated by
common monotonic trend (both length and retention change over time).

Zero GPU. Run: uv run python scripts/crosslag_robustness.py
"""
import json, re, math
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

CONFIGS = [
    ("g1_gen10_seed15", "qwen r=16", "homeostatic",
     {0: 1.0, 1: .949, 2: .949, 3: .949, 4: .949, 5: .949, 6: .949, 7: .949, 8: .949, 9: .949, 10: .949}),
    ("g1_rank256_seed15", "qwen r=256", "degradative",
     {0: 1.0, 1: .936, 2: .923, 3: .897, 4: .872, 5: .831, 6: .808, 7: .795, 8: .782, 9: .782, 10: .780}),
    ("g1_rank128_seed15", "qwen r=128", "bounded",
     {0: 1.0, 1: .910, 2: .897, 3: .885, 4: .872, 5: .873, 6: .885, 7: .885, 8: .885, 9: .885, 10: .886}),
    ("g1_rank64_seed15", "qwen r=64", "homeostatic",
     {0: 1.0, 1: .923, 2: .910, 3: .910, 4: .897, 5: .899, 6: .910, 7: .910, 8: .910, 9: .910, 10: .911}),
    ("gemma3_rank4_seed15", "gemma3 r=4", "homeostatic",
     {0: 1.0, 1: .936, 2: .936, 3: .936, 4: .923, 5: .922, 6: .922, 7: .936, 8: .936, 9: .936, 10: .936}),
    ("gemma3_rank16_seed15", "gemma3 r=16", "degradative",
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


def mean_length(responses):
    return sum(len(r.split()) for r in responses) / len(responses) if responses else 0


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    return num / den if den > 1e-10 else 0.0


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    rxy = pearson(x, y)
    rxz = pearson(x, z)
    ryz = pearson(y, z)
    denom = ((1 - rxz**2) * (1 - ryz**2)) ** 0.5
    if denom < 1e-10:
        return float('nan')
    return (rxy - rxz * ryz) / denom


def main():
    print("=" * 80)
    print("CROSS-LAG ROBUSTNESS CHECKS")
    print("=" * 80)

    for dir_name, label, regime, retention_map in CONFIGS:
        config_dir = OUTPUT_DIR / dir_name
        if not config_dir.exists():
            continue

        # Load mean_length per generation
        gen_lengths = {}
        for gen in range(0, 11):
            syn_file = config_dir / f"synthetic_gen{gen}.json"
            if syn_file.exists():
                data = json.load(open(syn_file))
                responses = extract_responses(data)
                gen_lengths[gen] = mean_length(responses)

        gens = sorted(set(gen_lengths.keys()) & set(retention_map.keys()))
        if len(gens) < 4:
            continue

        print(f"\n{'='*60}")
        print(f"  {label} ({regime})")
        print(f"{'='*60}")

        # Build aligned series
        lengths = [gen_lengths[g] for g in gens]
        retentions = [retention_map[g] for g in gens]

        # === CHECK 1: First-difference correlation ===
        # delta_length_t = length_t - length_{t-1}
        # delta_retention_{t+1} = retention_{t+1} - retention_t
        delta_len = [lengths[i] - lengths[i-1] for i in range(1, len(gens)-1)]
        delta_ret = [retentions[i+1] - retentions[i] for i in range(1, len(gens)-1)]

        r_diff = pearson(delta_len, delta_ret)
        print(f"\n  CHECK 1 - First-difference correlation:")
        print(f"    corr(delta_length_t, delta_retention_t+1) = {r_diff:+.3f}")
        print(f"    n = {len(delta_len)} pairs")

        # === CHECK 2: Partial correlation controlling for generation ===
        # corr(length_t, retention_t+1 | generation_t)
        len_t = lengths[:-1]
        ret_t1 = retentions[1:]
        gen_t = [float(g) for g in gens[:-1]]

        r_partial = partial_corr(len_t, ret_t1, gen_t)
        r_raw = pearson(len_t, ret_t1)
        print(f"\n  CHECK 2 - Partial correlation (controlling for generation):")
        print(f"    raw corr(length_t, retention_t+1) = {r_raw:+.3f}")
        print(f"    partial corr(length_t, retention_t+1 | gen_t) = {r_partial:+.3f}")
        print(f"    n = {len(len_t)} pairs")

        # === CHECK 3: Reverse-lag comparison ===
        # Forward: length_t -> retention_t+1
        # Reverse: retention_t -> length_t+1
        forward = pearson(lengths[:-1], retentions[1:])
        reverse = pearson(retentions[:-1], lengths[1:])
        print(f"\n  CHECK 3 - Reverse-lag comparison:")
        print(f"    FORWARD: corr(length_t, retention_t+1) = {forward:+.3f}")
        print(f"    REVERSE: corr(retention_t, length_t+1) = {reverse:+.3f}")
        diff = abs(forward) - abs(reverse)
        if diff > 0.05:
            print(f"    --> Forward stronger by {diff:.3f} (supports length -> retention)")
        elif diff < -0.05:
            print(f"    --> Reverse stronger by {-diff:.3f} (supports retention -> length)")
        else:
            print(f"    --> Similar magnitude (bidirectional / common trend)")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
  If Check 1 (first-diff) is negative: changes in length predict changes in retention.
    This is the strongest evidence for a directional mechanism.

  If Check 2 (partial) remains strong: the relationship holds beyond shared time trend.
    If it drops to ~0: the correlation was just "both change over time."

  If Check 3 (forward > reverse): temporal ordering favors length -> retention.
    If equal: bidirectional or common cause.
""")


if __name__ == "__main__":
    main()
