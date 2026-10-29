import json
from pathlib import Path

base = Path("Paradoxo/llm-knowledge-collapse/outputs")

print("=== RANK x LR MATRIX DETAILS ===")
f = base / "rank_lr_matrix"
for jf in sorted(f.glob("*.json")):
    data = json.load(open(jf, encoding="utf-8"))
    k0 = data.get("k0_size")
    gens = data.get("generations", [])
    rets = [f"G{g['gen']}:{g['retention']}/{k0}({g['retention']/k0*100:.1f}%)" for g in gens]
    print(f"  {jf.stem:<20} K0={k0}: {rets}")

print("\n=== FFT LR SWEEP QWEN (Gen 1-3) ===")
f = base / "fft_lr_sweep"
for jf in sorted(f.glob("fft_lr*.json")):
    data = json.load(open(jf, encoding="utf-8"))
    rets = [f"G{d['gen']}:{d['retention']} (abs={d['abs_drift']:.4f}, rel={d['rel_drift']:.4f})" for d in data]
    print(f"  {jf.stem:<15}: {rets}")
if (f / "qlora_r16.json").exists():
    data = json.load(open(f / "qlora_r16.json", encoding="utf-8"))
    rets = [f"G{d['gen']}:{d['retention']} (eff_rank={d['eff_rank']:.2f}, lora_norm={d['lora_norm']:.4f})" for d in data]
    print(f"  qlora_r16      : {rets}")

print("\n=== C1 BASELINE ON QWEN r=256 (K0=78) ===")
for jf in [base / "fft_lr_sweep" / "k0_indices.json"]:
    if jf.exists():
        k0_idx = json.load(open(jf))
        print("  k0_indices count:", len(k0_idx))
