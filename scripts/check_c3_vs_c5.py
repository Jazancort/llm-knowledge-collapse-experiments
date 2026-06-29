"""Final checks: C3 vs C5 overlap, token decomposition, training dynamics.

1. Which examples did C3 and C5 remove? Same or different?
2. Prompt tokens vs response tokens in each condition
3. Does the intervention reduce loss-bearing signal or just total sequence length?

Zero GPU. Run: uv run python scripts/check_c3_vs_c5.py
"""
import json, re, math
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

def extract_response(text):
    matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
    return matches[-1].strip() if matches else text.strip().split("\n")[-1].strip()

def main():
    print("=" * 80)
    print("C3 vs C5 ANALYSIS — What exactly is each intervention doing?")
    print("=" * 80)

    # Load Gen0 synthetic (shared starting point)
    gen0_path = OUTPUT_DIR / "causal_intervention" / "synthetic_gen0.json"
    if not gen0_path.exists():
        gen0_path = OUTPUT_DIR / "g1_rank256_seed15" / "synthetic_gen0.json"
    gen0 = json.load(open(gen0_path))
    n = len(gen0)

    # === Part 1: What does C3 filtering remove from Gen0? ===
    print(f"\n--- PART 1: What does C3 remove vs C5? ---")
    print(f"  Gen0 has {n} examples")

    # Simulate C3 filter (same logic as causal_intervention.py)
    c3_kept = []
    c3_removed = []
    for i, text in enumerate(gen0):
        response = extract_response(text)
        words = response.split()
        if 1 <= len(words) <= 5:
            c3_kept.append(i)
        else:
            c3_removed.append(i)

    print(f"  C3 keeps {len(c3_kept)} examples, removes {len(c3_removed)}")

    # Simulate C5 random downsample (target ~54500 tokens)
    target_tokens = 54500
    np.random.seed(15)  # Same seed as validation script
    indices = list(range(n))
    np.random.shuffle(indices)
    c5_kept = []
    total = 0
    for idx in indices:
        tokens = len(gen0[idx].split())
        if total + tokens > target_tokens:
            break
        c5_kept.append(idx)
        total += tokens
    c5_removed = [i for i in range(n) if i not in set(c5_kept)]

    print(f"  C5 keeps {len(c5_kept)} examples, removes {len(c5_removed)}")

    # Overlap
    c3_rem_set = set(c3_removed)
    c5_rem_set = set(c5_removed)
    overlap = c3_rem_set & c5_rem_set
    jaccard = len(overlap) / len(c3_rem_set | c5_rem_set) if (c3_rem_set | c5_rem_set) else 0
    print(f"\n  Overlap of removed examples:")
    print(f"    C3 removed: {len(c3_removed)}")
    print(f"    C5 removed: {len(c5_removed)}")
    print(f"    Shared removed: {len(overlap)}")
    print(f"    Jaccard: {jaccard:.3f}")

    # Length of removed examples
    c3_rem_lengths = [len(extract_response(gen0[i]).split()) for i in c3_removed]
    c5_rem_lengths = [len(extract_response(gen0[i]).split()) for i in c5_removed]
    c3_kept_lengths = [len(extract_response(gen0[i]).split()) for i in c3_kept]

    print(f"\n  Response lengths of removed examples:")
    print(f"    C3 removed avg: {sum(c3_rem_lengths)/len(c3_rem_lengths):.1f} words (all >5 by definition)")
    if c5_rem_lengths:
        print(f"    C5 removed avg: {sum(c5_rem_lengths)/len(c5_rem_lengths):.1f} words (random)")
    print(f"    C3 kept avg: {sum(c3_kept_lengths)/len(c3_kept_lengths):.1f} words")

    # === Part 2: Token decomposition ===
    print(f"\n--- PART 2: Prompt vs Response tokens ---")

    # For each condition's synthetic data, decompose into prompt and response tokens
    conditions = {
        "C1_normal_gen1": OUTPUT_DIR / "g1_rank256_seed15" / "synthetic_gen1.json",
        "C3_filtered_gen1": OUTPUT_DIR / "causal_intervention" / "syn_C3_length_filtered_gen1.json",
        "C5_matched_gen1": OUTPUT_DIR / "intervention_validation" / "syn_C5_token_matched_s15_gen1.json",
    }

    print(f"\n  {'Condition':<25} {'N_ex':>6} {'Tot_tok':>8} {'Prompt_tok':>10} {'Resp_tok':>9} {'Resp%':>6}")
    print(f"  {'-'*65}")

    for name, path in conditions.items():
        if not path.exists():
            print(f"  {name:<25} FILE NOT FOUND: {path.name}")
            continue
        data = json.load(open(path))
        total_tokens = 0
        prompt_tokens = 0
        resp_tokens = 0
        for text in data:
            all_tok = len(text.split())
            total_tokens += all_tok
            resp = extract_response(text)
            r_tok = len(resp.split())
            resp_tokens += r_tok
            prompt_tokens += (all_tok - r_tok)
        resp_pct = resp_tokens / total_tokens * 100 if total_tokens > 0 else 0
        print(f"  {name:<25} {len(data):>6} {total_tokens:>8} {prompt_tokens:>10} {resp_tokens:>9} {resp_pct:>5.1f}%")

    # === Part 3: Critical boundary analysis ===
    print(f"\n--- PART 3: How many examples tip the balance? ---")

    # C1 has 2000 examples. C3 keeps ~X. C5 keeps ~Y.
    # Both get +9pp. So removing ~100 examples = +9pp.
    diff_examples = n - len(c3_kept)
    print(f"\n  C1: {n} examples -> 65/78 retention")
    print(f"  C3: {len(c3_kept)} examples (removed {diff_examples}) -> 73/78 retention")
    print(f"  C5: {len(c5_kept)} examples (removed {n - len(c5_kept)}) -> 72/78 retention")
    print(f"\n  Removing ~{diff_examples} examples ({diff_examples/n*100:.1f}% of dataset) = +8-9pp retention")
    print(f"  That's ~{(73-65)/diff_examples:.1f} pp per 100 removed examples")
    print(f"\n  CONCLUSION: The degradative regime is critically balanced.")
    print(f"  ~5% reduction in synthetic training examples shifts from degradative to near-homeostatic.")

    # === Part 4: Are C3-removed examples the "problematic" ones? ===
    print(f"\n--- PART 4: Are long responses actually worse? ---")

    # Check: in Gen0, are items with longer responses the ones that later become errors?
    # We can't directly test this without k0_results per item in C1, but we can check:
    # - What % of C3-removed items are factoid (short correct) vs verbose (long, possibly wrong)?
    gen0_responses = [extract_response(t) for t in gen0]
    all_lengths = [len(r.split()) for r in gen0_responses]

    short = [l for l in all_lengths if l <= 5]
    long_resp = [l for l in all_lengths if l > 5]

    print(f"\n  Gen0 response length distribution:")
    print(f"    <= 5 words: {len(short)} ({len(short)/n*100:.1f}%)")
    print(f"    > 5 words:  {len(long_resp)} ({len(long_resp)/n*100:.1f}%)")
    print(f"    Mean: {sum(all_lengths)/len(all_lengths):.1f}, Median: {sorted(all_lengths)[n//2]}")

    # Length histogram
    buckets = {1:0, 2:0, 3:0, 4:0, 5:0, "6-10":0, "11+":0}
    for l in all_lengths:
        if l <= 5:
            buckets[l] = buckets.get(l, 0) + 1
        elif l <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11+"] += 1
    print(f"\n  Length histogram:")
    for k, v in buckets.items():
        bar = "#" * (v // 20)
        print(f"    {str(k):>5} words: {v:>5} ({v/n*100:.1f}%) {bar}")


if __name__ == "__main__":
    main()
