"""Analyze all v3 experiment results."""
import json
from pathlib import Path

output_lines = []

for f in sorted(Path("outputs").glob("v3_*/results.json")):
    d = json.loads(f.read_text())
    key = f.parent.name
    model = d["model"].split("/")[-1]
    rank = d["rank"]
    seed = d["seed"]
    k0 = d["k0_size"]
    gens = d.get("generations", [])

    output_lines.append(f"=== {key} ===")
    output_lines.append(f"  Model: {model}, r={rank}, seed={seed}, K0={k0}, Gens: {len(gens)}")

    for g in gens:
        ret = g["retention"]
        pct = ret / k0 * 100
        dist = g.get("distributional", {})
        ml = dist.get("mean_length_words", "?")
        d1 = dist.get("distinct_1", "?")
        sw = dist.get("stopword_ratio", "?")
        er = g["eff_rank"]
        gen_num = g["gen"]
        output_lines.append(f"    Gen{gen_num:2d}: {ret}/{k0} ({pct:5.1f}%) erank={er:.2f} len={ml}w d1={d1} sw={sw}")

    output_lines.append("")

# Save
out_path = Path("outputs/v3_analysis.txt")
out_path.write_text("\n".join(output_lines), encoding="utf-8")
print(f"Saved to {out_path}")
print(f"Total experiments: {len(list(Path('outputs').glob('v3_*/results.json')))}")
