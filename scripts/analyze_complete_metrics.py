"""Complete metrics suite: everything GPT requested that wasn't implemented yet.

Adds to the existing diversity analysis:
1. MTLD (Moving-Average Type-Token Ratio) — length-robust diversity
2. Stopword/content ratio — detects "filler" verbosity
3. Content efficiency — retention / mean_length (factual density)
4. Normalized Levenshtein item stability — fuzzy stability beyond token-F1
5. SDI with 3 terms (length + distinct + instability)
6. Wrong-answer persistence — same wrong answer recurring across gens per item

Zero GPU. Run: uv run python scripts/analyze_complete_metrics.py
"""
import json, re, math
from pathlib import Path
from collections import Counter
import csv

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "diversity_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

STOPWORDS = {"the","a","an","is","are","was","were","be","been","being","have","has","had",
             "do","does","did","will","would","could","should","may","might","shall","can",
             "it","its","this","that","these","those","i","you","he","she","we","they",
             "my","your","his","her","our","their","me","him","us","them",
             "in","on","at","to","for","of","with","by","from","as","into","through",
             "and","or","but","if","so","yet","not","no","very","just","also","then"}

CONFIGS = [
    ("g1_gen10_seed15", "qwen", "r16", "homeostatic", 10,
     {0:1.0,1:.949,2:.949,3:.949,4:.949,5:.949,6:.949,7:.949,8:.949,9:.949,10:.949}),
    ("g1_rank256_seed15", "qwen", "r256", "degradative", 10,
     {0:1.0,1:.936,2:.923,3:.897,4:.872,5:.831,6:.808,7:.795,8:.782,9:.782,10:.780}),
    ("g1_rank128_seed15", "qwen", "r128", "bounded", 10,
     {0:1.0,1:.910,2:.897,3:.885,4:.872,5:.873,6:.885,7:.885,8:.885,9:.885,10:.886}),
    ("gemma3_rank4_seed15", "gemma3", "r4", "homeostatic", 10,
     {0:1.0,1:.936,2:.936,3:.936,4:.923,5:.922,6:.922,7:.936,8:.936,9:.936,10:.936}),
    ("gemma3_rank16_seed15", "gemma3", "r16", "degradative", 5,
     {0:1.0,1:.846,2:.756,3:.718,4:.692,5:.688}),
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


def compute_mtld(tokens, threshold=0.72):
    """Moving-Average Type-Token Ratio — robust to length."""
    if len(tokens) < 10:
        return 0.0
    factors = 0.0
    factor_len = 0
    types = set()
    for token in tokens:
        types.add(token)
        factor_len += 1
        ttr = len(types) / factor_len
        if ttr <= threshold:
            factors += 1
            types = set()
            factor_len = 0
    # Partial factor
    if factor_len > 0:
        ttr = len(types) / factor_len
        factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
    return len(tokens) / factors if factors > 0 else len(tokens)


def compute_stopword_ratio(tokens):
    """Proportion of tokens that are stopwords (filler detection)."""
    if not tokens:
        return 0.0
    stop_count = sum(1 for t in tokens if t in STOPWORDS)
    return stop_count / len(tokens)


def levenshtein_distance(s1, s2):
    """Normalized Levenshtein distance (0=identical, 1=completely different)."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    prev_row = list(range(len2 + 1))
    for i in range(len1):
        curr_row = [i + 1]
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[len2] / max(len1, len2)


def normalize_for_comparison(text):
    """Normalize text for fuzzy comparison."""
    text = text.lower().strip()
    for prefix in ["the answer is ", "it is ", "that would be ", "i believe it's ",
                   "it's ", "that's "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    print("=" * 80)
    print("COMPLETE METRICS SUITE")
    print("=" * 80)

    all_rows = []

    for dir_name, model, rank, regime, max_gen, retention_map in CONFIGS:
        config_dir = OUTPUT_DIR / dir_name
        if not config_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  {dir_name} ({regime})")
        print(f"{'='*60}")

        # Load all generations
        gen_data = {}
        for gen in range(0, max_gen + 1):
            syn_file = config_dir / f"synthetic_gen{gen}.json"
            if syn_file.exists():
                raw = json.load(open(syn_file))
                gen_data[gen] = extract_responses(raw)
        
        if not gen_data:
            continue

        gens = sorted(gen_data.keys())
        gen0_responses = gen_data[gens[0]]
        gen0_lengths = [len(r.split()) for r in gen0_responses]
        gen0_mean_len = sum(gen0_lengths) / len(gen0_lengths)

        print(f"\n  {'Gen':<4} {'d1':>5} {'MTLD':>6} {'stop%':>6} {'len':>5} "
              f"{'cont_eff':>8} {'lev_stab':>8} {'persist':>7} {'ret':>5}")
        print(f"  {'-'*65}")

        prev_responses = None

        for gen in gens:
            responses = gen_data[gen]
            all_tokens = [t for r in responses for t in r.lower().split()]
            total_tokens = len(all_tokens)

            # Standard
            distinct_1 = len(set(all_tokens)) / total_tokens if total_tokens > 0 else 0

            # MTLD
            mtld = compute_mtld(all_tokens)

            # Stopword ratio
            stopword_ratio = compute_stopword_ratio(all_tokens)

            # Mean length
            lengths = [len(r.split()) for r in responses]
            mean_length = sum(lengths) / len(lengths)

            # Content efficiency: retention / mean_length
            ret = retention_map.get(gen, None)
            content_eff = ret / mean_length if ret and mean_length > 0 else None

            # Levenshtein stability vs previous gen
            lev_stability = None
            if prev_responses and len(prev_responses) == len(responses):
                lev_dists = []
                for a, b in zip(prev_responses[:500], responses[:500]):  # sample 500 for speed
                    na = normalize_for_comparison(a)
                    nb = normalize_for_comparison(b)
                    lev_dists.append(1 - levenshtein_distance(na, nb))  # 1 = identical
                lev_stability = sum(lev_dists) / len(lev_dists)

            # Wrong-answer persistence: same response as Gen0 for same item
            persistence = None
            if gen > 0 and len(gen0_responses) == len(responses):
                same_as_gen0 = sum(
                    1 for a, b in zip(gen0_responses[:1000], responses[:1000])
                    if normalize_for_comparison(a) == normalize_for_comparison(b)
                ) / min(1000, len(responses))
                persistence = same_as_gen0

            # Print
            ret_str = f"{ret:.3f}" if ret else "  -"
            ce_str = f"{content_eff:.4f}" if content_eff else "   -"
            ls_str = f"{lev_stability:.3f}" if lev_stability is not None else "   -"
            ps_str = f"{persistence:.3f}" if persistence is not None else "   -"

            print(f"  {gen:<4} {distinct_1:>5.3f} {mtld:>6.1f} {stopword_ratio:>6.3f} "
                  f"{mean_length:>5.1f} {ce_str:>8} {ls_str:>8} {ps_str:>7} {ret_str:>5}")

            all_rows.append({
                "config": dir_name, "model": model, "rank": rank,
                "regime": regime, "gen": gen, "retention": ret,
                "distinct_1": round(distinct_1, 4),
                "mtld": round(mtld, 1),
                "stopword_ratio": round(stopword_ratio, 4),
                "mean_length": round(mean_length, 2),
                "content_efficiency": round(content_eff, 4) if content_eff else None,
                "levenshtein_stability": round(lev_stability, 4) if lev_stability is not None else None,
                "gen0_persistence": round(persistence, 4) if persistence is not None else None,
            })

            prev_responses = responses

        # SDI with 3 terms
        if len(gens) >= 2:
            first_metrics = all_rows[-len(gens)]
            last_metrics = all_rows[-1]
            lr = last_metrics["mean_length"] / max(first_metrics["mean_length"], 0.1)
            dr = first_metrics["distinct_1"] / max(last_metrics["distinct_1"], 0.001)
            # Item instability: 1 - levenshtein_stability of last gen
            instab = 1 - (last_metrics["levenshtein_stability"] or 0.5)
            instab_first = 1 - (all_rows[-len(gens)+1].get("levenshtein_stability") or 0.5) if len(gens) > 1 else instab
            instab_increase = max(instab - instab_first, 0)
            sdi3 = math.log(lr) + math.log(dr) + instab_increase
            print(f"\n  SDI-3 (Gen0 vs Gen{gens[-1]}): {sdi3:.3f}")
            print(f"    log(length_ratio)={math.log(lr):.3f} + log(d1_ratio)={math.log(dr):.3f} + instab_increase={instab_increase:.3f}")

    # Save
    csv_path = RESULTS_DIR / "complete_metrics.csv"
    if all_rows:
        keys = all_rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Saved: {csv_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
