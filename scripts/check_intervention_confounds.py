"""Confound checks for causal intervention: token budget + item-level analysis.

Checks whether the retention improvement from C2/C3/C4 is due to:
  (a) better data quality, or
  (b) simply fewer tokens / smaller update magnitude

Also: which specific facts were rescued by interventions?

Zero GPU. Run: uv run python scripts/check_intervention_confounds.py
"""
import json, re
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
INTERVENTION_DIR = OUTPUT_DIR / "causal_intervention"

# Baseline retention per gen (from g1_rank256_seed15)
C1_RETENTION = [73, 72, 70, 68, 65]


def extract_responses(texts):
    responses = []
    for text in texts:
        matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
        if matches:
            responses.append(matches[-1].strip())
        else:
            responses.append(text.strip().split("\n")[-1].strip())
    return responses


def count_tokens_in_synthetic(synthetic_texts):
    """Count total training tokens (full sequences, not just responses)."""
    total_tokens = 0
    for text in synthetic_texts:
        total_tokens += len(text.split())
    return total_tokens


def main():
    print("=" * 80)
    print("CONFOUND CHECKS — TOKEN BUDGET + ITEM-LEVEL")
    print("=" * 80)

    # === CHECK 1: Token budget per condition per generation ===
    print("\n--- CHECK 1: TOKEN BUDGET ---")
    print(f"\n  {'Condition':<25} {'Gen':<5} {'Examples':>8} {'Tokens':>8} {'Tok/ex':>7} {'Resp_len':>8}")
    print(f"  {'-'*65}")

    conditions = {
        "C1_normal": ("g1_rank256_seed15", "synthetic_gen"),
        "C2_short": ("causal_intervention", "syn_C2_short_constrained_gen"),
        "C3_filtered": ("causal_intervention", "syn_C3_length_filtered_gen"),
        "C4_canonical": ("causal_intervention", "syn_C4_canonical_gen"),
    }

    token_budgets = {}

    for cond_name, (subdir, prefix) in conditions.items():
        cond_dir = OUTPUT_DIR / subdir
        token_budgets[cond_name] = []

        for gen in range(1, 6):
            # The synthetic used to TRAIN gen N is from gen N-1
            train_gen = gen - 1
            syn_file = cond_dir / f"{prefix}{train_gen}.json"
            if not syn_file.exists():
                # Try shared gen0
                if train_gen == 0:
                    syn_file = INTERVENTION_DIR / "synthetic_gen0.json"
                    if not syn_file.exists():
                        syn_file = cond_dir / "synthetic_gen0.json"
                if not syn_file.exists():
                    continue

            data = json.load(open(syn_file))
            n_examples = len(data)
            total_tokens = count_tokens_in_synthetic(data)
            tok_per_ex = total_tokens / n_examples if n_examples > 0 else 0

            responses = extract_responses(data)
            resp_lengths = [len(r.split()) for r in responses]
            mean_resp_len = sum(resp_lengths) / len(resp_lengths) if resp_lengths else 0

            token_budgets[cond_name].append({
                "gen": gen, "examples": n_examples, "tokens": total_tokens,
                "tok_per_ex": tok_per_ex, "mean_resp_len": mean_resp_len
            })

            print(f"  {cond_name:<25} {gen:<5} {n_examples:>8} {total_tokens:>8} "
                  f"{tok_per_ex:>7.1f} {mean_resp_len:>8.1f}")

    # Summary: average token budget
    print(f"\n  SUMMARY (average across gens):")
    print(f"  {'Condition':<25} {'Avg tokens':>10} {'Avg tok/ex':>10} {'Avg resp_len':>12} {'Gen5 ret':>8}")
    print(f"  {'-'*70}")

    gen5_rets = {"C1_normal": 65, "C2_short": 71, "C3_filtered": 73, "C4_canonical": 69}

    for cond_name in conditions:
        if token_budgets[cond_name]:
            avg_tok = sum(t["tokens"] for t in token_budgets[cond_name]) / len(token_budgets[cond_name])
            avg_tpe = sum(t["tok_per_ex"] for t in token_budgets[cond_name]) / len(token_budgets[cond_name])
            avg_rlen = sum(t["mean_resp_len"] for t in token_budgets[cond_name]) / len(token_budgets[cond_name])
            ret = gen5_rets.get(cond_name, "?")
            print(f"  {cond_name:<25} {avg_tok:>10.0f} {avg_tpe:>10.1f} {avg_rlen:>12.1f} {ret:>8}")

    # === CHECK 2: Item-level — which facts were rescued? ===
    print("\n\n--- CHECK 2: ITEM-LEVEL FACT RESCUE ---")

    # Load k0_results for each condition
    results_by_cond = {}
    for cond in ["C2_short_constrained", "C3_length_filtered", "C4_canonical"]:
        f = INTERVENTION_DIR / f"{cond}.json"
        if f.exists():
            data = json.load(open(f))
            # Gen5 k0_results
            if len(data) >= 5:
                results_by_cond[cond] = data[4]["k0_results"]  # Gen5

    # C1 baseline Gen5 k0_results (from g1_rank256_seed15 if available)
    c1_gen5_path = OUTPUT_DIR / "g1_rank256_seed15" / "results.json"
    c1_k0 = None
    # We don't have per-item k0 for C1 in the old format; use the Gen10 data
    gen10_path = OUTPUT_DIR / "fft_drift_gen10" / "fft_lr1e-06_seed15.json"  # wrong, need rank256
    # Actually let's compute from the intervention's Gen1 shared start
    # All conditions start from same Gen0, so Gen1 should be similar

    # Compare conditions at Gen5
    if results_by_cond:
        print(f"\n  Facts lost at Gen5 (of K0=78):")
        for cond, k0_res in results_by_cond.items():
            lost = [i for i, r in enumerate(k0_res) if not r]
            print(f"    {cond}: lost {len(lost)} items: {lost}")

        # Cross-condition overlap
        if len(results_by_cond) >= 2:
            conds = list(results_by_cond.keys())
            print(f"\n  Overlap between conditions (facts lost):")
            for i, c1 in enumerate(conds):
                for c2 in conds[i+1:]:
                    lost1 = set(i for i, r in enumerate(results_by_cond[c1]) if not r)
                    lost2 = set(i for i, r in enumerate(results_by_cond[c2]) if not r)
                    shared = lost1 & lost2
                    print(f"    {c1} vs {c2}: shared={sorted(shared)}")

            # Facts that ALL interventions preserve but baseline loses
            all_preserved = set(range(78))
            for cond, k0_res in results_by_cond.items():
                preserved = set(i for i, r in enumerate(k0_res) if r)
                all_preserved &= preserved

            # Compare Gen1 (all same) vs Gen5 (diverge)
            gen1_lost = set(i for i, r in enumerate(results_by_cond[conds[0]]) if not r)
            # Actually Gen1 k0 is in index 0 of each condition
            for cond in conds:
                f = INTERVENTION_DIR / f"{cond}.json"
                data = json.load(open(f))
                gen1_k0 = data[0]["k0_results"]
                gen1_lost_items = [i for i, r in enumerate(gen1_k0) if not r]
                gen5_lost_items = [i for i, r in enumerate(results_by_cond[cond]) if not r]
                new_losses = set(gen5_lost_items) - set(gen1_lost_items)
                print(f"\n    {cond}:")
                print(f"      Gen1 lost: {gen1_lost_items}")
                print(f"      Gen5 lost: {gen5_lost_items}")
                print(f"      New losses Gen1->Gen5: {sorted(new_losses)} ({len(new_losses)} items)")

    # === INTERPRETATION ===
    print("\n\n--- INTERPRETATION ---")
    print("""
  TOKEN BUDGET:
    If C2/C3 have significantly fewer tokens than C1:
      Part of the gain may come from reduced update magnitude.
    If tokens are similar:
      Gain is from data quality/format.

  ITEM-LEVEL:
    If all interventions rescue the same facts:
      Those facts are universally fragile to verbosity.
    If different facts are rescued:
      Each intervention protects via different mechanism.
""")


if __name__ == "__main__":
    main()
