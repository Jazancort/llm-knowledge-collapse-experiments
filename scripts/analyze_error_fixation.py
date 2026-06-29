"""Analysis C: Error fixation — do degradative models repeat the same wrong answers?

Checks whether wrong answers become fixed (ossified) across generations,
or whether the model cycles through different errors.

Zero GPU. Analyzes response text from synthetic JSONs.

Run: uv run python scripts/analyze_error_fixation.py
"""
import json, re
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "diversity_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

# Configs with multiple generations to compare item-level answers
CONFIGS = [
    ("g1_gen10_seed15", "qwen", "r16", "homeostatic", 10),
    ("g1_rank256_seed15", "qwen", "r256", "degradative", 10),
    ("g1_rank128_seed15", "qwen", "r128", "bounded", 10),
    ("gemma3_rank4_seed15", "gemma3", "r4", "homeostatic", 10),
    ("gemma3_rank16_seed15", "gemma3", "r16", "degradative", 5),
    ("fft_comparison_fft_seed15", "qwen_fft", "lr2e-6", "homeostatic", 5),
]


def extract_responses(texts):
    """Extract assistant responses from chat-template formatted synthetic data."""
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


def main():
    print("=" * 80)
    print("ERROR FIXATION ANALYSIS")
    print("=" * 80)
    print("\nQuestion: Do models repeat the same answers across generations (ossification)?")
    print("Or do answers change every generation (flux)?")

    for dir_name, model, rank_lr, regime, max_gen in CONFIGS:
        config_dir = OUTPUT_DIR / dir_name
        if not config_dir.exists():
            continue

        print(f"\n{'='*70}")
        print(f"  {dir_name} ({regime})")
        print(f"{'='*70}")

        # Load responses per generation (by item index)
        gen_responses = {}
        for gen in range(0, max_gen + 1):
            syn_file = config_dir / f"synthetic_gen{gen}.json"
            if not syn_file.exists():
                continue
            data = json.load(open(syn_file))
            responses = extract_responses(data)
            gen_responses[gen] = responses

        if len(gen_responses) < 3:
            print("    Not enough generations, skipping.")
            continue

        gens = sorted(gen_responses.keys())
        n_items = min(len(gen_responses[g]) for g in gens)

        # Measure item-level stability: how often does the answer for item i
        # stay the same between consecutive generations?
        stability_per_gen = []
        for idx in range(len(gens) - 1):
            g1, g2 = gens[idx], gens[idx + 1]
            same_count = 0
            for i in range(n_items):
                if gen_responses[g1][i].lower().strip() == gen_responses[g2][i].lower().strip():
                    same_count += 1
            stability = same_count / n_items
            stability_per_gen.append((g1, g2, stability))

        print(f"\n  Item-level stability (same answer between consecutive gens):")
        for g1, g2, stab in stability_per_gen:
            bar = "#" * int(stab * 40)
            print(f"    Gen{g1}->Gen{g2}: {stab:.3f} ({stab*100:.1f}%) {bar}")

        # Overall: Gen0 vs GenFinal — how many items give same answer?
        g_first = gens[0]
        g_last = gens[-1]
        same_first_last = sum(
            1 for i in range(n_items)
            if gen_responses[g_first][i].lower().strip() == gen_responses[g_last][i].lower().strip()
        ) / n_items
        print(f"\n  Gen{g_first} vs Gen{g_last} (same answer): {same_first_last:.3f} ({same_first_last*100:.1f}%)")

        # How many items NEVER change (stable from Gen0 to GenFinal)?
        never_change = 0
        for i in range(n_items):
            first_resp = gen_responses[gens[0]][i].lower().strip()
            all_same = all(gen_responses[g][i].lower().strip() == first_resp for g in gens)
            if all_same:
                never_change += 1
        print(f"  Items that NEVER change: {never_change}/{n_items} ({never_change/n_items*100:.1f}%)")

        # How many items change in Gen0->Gen1 but then freeze?
        if len(gens) >= 3:
            changed_g1 = 0
            frozen_after = 0
            for i in range(n_items):
                r0 = gen_responses[gens[0]][i].lower().strip()
                r1 = gen_responses[gens[1]][i].lower().strip()
                if r0 != r1:
                    changed_g1 += 1
                    # Did it stay the same from Gen1 onward?
                    r1_frozen = all(
                        gen_responses[g][i].lower().strip() == r1
                        for g in gens[2:]
                    )
                    if r1_frozen:
                        frozen_after += 1
            print(f"  Items changed at Gen1: {changed_g1}/{n_items} ({changed_g1/n_items*100:.1f}%)")
            if changed_g1 > 0:
                print(f"  Of those, frozen after Gen1: {frozen_after}/{changed_g1} ({frozen_after/changed_g1*100:.1f}%)")

        # Top repeated answer (mode collapse indicator)
        last_gen = gen_responses[g_last]
        answer_counts = Counter(r.lower().strip() for r in last_gen)
        top5 = answer_counts.most_common(5)
        print(f"\n  Top-5 most repeated answers in Gen{g_last}:")
        for ans, count in top5:
            print(f"    '{ans[:50]}' : {count}x ({count/n_items*100:.1f}%)")


if __name__ == "__main__":
    main()
