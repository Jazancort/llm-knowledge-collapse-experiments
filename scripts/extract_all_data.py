import json
from pathlib import Path

base = Path("Paradoxo/llm-knowledge-collapse/outputs")

print("======================================================================")
print("1. QWEN RANK SWEEP (seed 15)")
print("======================================================================")
for r in [4, 16, 32, 64, 128, 256]:
    f = base / f"g1_rank{r}_seed15/results.json"
    if not f.exists() and r == 16:
        f = base / "g1_gen10_seed15/results.json"
    if f.exists():
        data = json.load(open(f, encoding="utf-8"))
        k0 = data[0].get("k0_size")
        rets = [f"G{d['generation']}:{d['retention']}/{k0}({d['retention']/k0*100:.1f}%)" for d in data]
        eranks = [f"{d['adapter']['eff_rank']:.2f}" if d.get("adapter") else "-" for d in data]
        transitions = [d.get("transitions") for d in data]
        print(f"r={r:<3} K0={k0}:")
        print(f"  Retention: {rets}")
        print(f"  Eff Ranks: {eranks}")
        print(f"  Transitions: {transitions}")

print("\n======================================================================")
print("2. QWEN r=256 MULTI-SEED (seeds 15, 137, 256)")
print("======================================================================")
for s in [15, 137, 256]:
    f = base / f"g1_rank256_seed{s}/results.json"
    if f.exists():
        data = json.load(open(f, encoding="utf-8"))
        k0 = data[0].get("k0_size")
        rets = [f"G{d['generation']}:{d['retention']}/{k0}({d['retention']/k0*100:.1f}%)" for d in data]
        print(f"seed {s:<4} K0={k0}: {rets}")

print("\n======================================================================")
print("3. QWEN FULL LINEAR (r=4, r=16)")
print("======================================================================")
for r in [4, 16]:
    f = base / f"g1_rank{r}_full_seed15/results.json"
    if f.exists():
        data = json.load(open(f, encoding="utf-8"))
        k0 = data[0].get("k0_size")
        rets = [f"G{d['generation']}:{d['retention']}/{k0}({d['retention']/k0*100:.1f}%)" for d in data]
        eranks = [f"{d['adapter']['eff_rank']:.2f}" if d.get("adapter") else "-" for d in data]
        print(f"full r={r} K0={k0}: {rets} | eranks={eranks}")

print("\n======================================================================")
print("4. FFT vs QLORA DRIFT MATCHED (seeds 15, 137, 256, 10 gens)")
print("======================================================================")
for method in ["fft_lr1e-06", "qlora_r16"]:
    print(f"\n--- {method} ---")
    for s in [15, 137, 256]:
        f = base / f"fft_drift_gen10/{method}_seed{s}.json"
        if f.exists():
            data = json.load(open(f, encoding="utf-8"))
            rets = [f"G{d['gen']}:{d['retention']}" for d in data]
            drifts = [f"{d.get('abs_drift', d.get('lora_ba_norm', d.get('lora_norm', '-')))}" for d in data]
            print(f"  seed {s:<4}: {rets}")
            print(f"         drift/norm: {drifts}")

print("\n======================================================================")
print("5. RANK x LR MATRIX (Qwen)")
print("======================================================================")
f = base / "rank_lr_matrix"
for jf in sorted(f.glob("*.json")):
    try:
        data = json.load(open(jf, encoding="utf-8"))
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            rets = [f"G{d.get('gen', d.get('generation'))}:{d.get('retention')}" for d in data]
            print(f"  {jf.stem:<25}: {rets}")
        elif isinstance(data, dict):
            print(f"  {jf.stem:<25}: {list(data.keys())[:5]}")
    except Exception as e:
        print(f"  {jf.stem:<25}: error {e}")

print("\n======================================================================")
print("6. GEMMA 3 INTERMEDIATE RANKS & FFT")
print("======================================================================")
for ir in [10, 12, 14]:
    f = base / f"gemma3_intermediate_ranks/gemma3_rank{ir}_seed15.json"
    if f.exists():
        data = json.load(open(f, encoding="utf-8"))
        k0 = data.get("k0_size")
        gens = data.get("generations", [])
        rets = [f"G{g['gen']}:{g['retention']}/{k0}({g['retention']/k0*100:.1f}%)" for g in gens]
        eranks = [f"{g['eff_rank']:.2f}" for g in gens]
        print(f"  Gemma 3 r={ir} K0={k0}: {rets} | eranks={eranks}")

for lr in ["1e-06", "5e-06", "1e-05"]:
    f = base / f"gemma3_fft/gemma3_fft_lr{lr}.json"
    if f.exists():
        data = json.load(open(f, encoding="utf-8"))
        k0 = data.get("k0_size")
        gens = data.get("generations", [])
        rets = [f"G{g['gen']}:{g['retention']}/{k0}({g['retention']/k0*100:.1f}%)" for g in gens]
        print(f"  Gemma 3 FFT LR={lr} K0={k0}: {rets}")

print("\n======================================================================")
print("7. INTERVENTIONS C1-C5 (Qwen r=256, K0=78)")
print("======================================================================")
for folder in ["causal_intervention", "intervention_validation", "c5_masks"]:
    p = base / folder
    print(f"\n--- {folder} ---")
    for jf in sorted(p.glob("*.json")):
        if "syn" in jf.name or "synthetic" in jf.name:
            continue
        try:
            data = json.load(open(jf, encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                rets = [f"G{d.get('gen', d.get('generation'))}:{d.get('retention')}" for d in data]
                print(f"  {jf.stem:<25}: {rets}")
        except Exception as e:
            print(f"  {jf.stem:<25}: error {e}")

