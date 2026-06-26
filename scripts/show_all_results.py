"""Display all recent experiment results."""
import json
from pathlib import Path

base = Path(__file__).parent.parent / "outputs"

print("=" * 70)
print("GEMMA 3 1B IT — SPRINT 2 RESULTS")
print("=" * 70)

for rank in [2, 4, 16, 256]:
    path = base / f"gemma3_rank{rank}_seed15/results.json"
    if not path.exists():
        continue
    data = json.load(open(path))
    k0 = data[0].get("k0_size", 47)
    print(f"\n--- Gemma r={rank} (K0={k0}) ---")
    print(f"  Gen  Retention     Eff.Rank      C->W  W->C")
    for r in data:
        gen = r["generation"]
        ret = r["retention"]
        pct = ret / k0 * 100
        adapter = r.get("adapter")
        er = f"{adapter['eff_rank']:.2f}/{rank}" if adapter else "-"
        trans = r.get("transitions")
        cw = trans["C->W"] if trans else "-"
        wc = trans["W->C"] if trans else "-"
        print(f"  {gen:<4} {ret}/{k0} ({pct:.1f}%)   {er:<14} {cw:<5} {wc}")

print("\n" + "=" * 70)
print("QWEN 2.5 1.5B — ALL CONFIGS WITH GEN10")
print("=" * 70)

qwen_configs = [
    ("Attn r=4", "g1_rank4_seed15"),
    ("Attn r=16", "g1_gen10_seed15"),
    ("Attn r=32", "g1_rank32_seed15"),
    ("Attn r=64", "g1_rank64_seed15"),
    ("Attn r=128", "g1_rank128_seed15"),
    ("Attn r=256", "g1_rank256_seed15"),
    ("Full r=4", "g1_rank4_full_seed15"),
    ("Full r=16", "g1_rank16_full_seed15"),
]

for name, folder in qwen_configs:
    path = base / folder / "results.json"
    if not path.exists():
        continue
    data = json.load(open(path))
    k0 = data[0].get("k0_size", 79)
    last = data[-1]
    last_gen = last["generation"]
    ret = last["retention"]
    pct = ret / k0 * 100
    adapter = last.get("adapter")
    er = f"{adapter['eff_rank']:.1f}" if adapter else "-"
    regime = "Homeostatic" if pct > 85 else ("Quasi-homeo" if pct > 82 else "Degradative")
    print(f"  {name:<15} Gen{last_gen:<3} {ret}/{k0} ({pct:.1f}%)  eff_rank={er:<6} {regime}")

print("\n" + "=" * 70)
print("CROSS-ARCHITECTURE COMPARISON")
print("=" * 70)
print("""
  Model      Rank  Eff.Rank  Gen5 Ret   Gen10 Ret  Regime
  -----------------------------------------------------------------
  Qwen       r=4   3.3       96.2%      -          Homeostatic
  Qwen       r=16  11.1      94.9%      94.9%      Homeostatic
  Qwen       r=32  17.8      94.9%      -          Homeostatic
  Qwen       r=64  30.1      89.9%      91.1%      Homeostatic
  Qwen       r=128 50.5      87.3%      88.6%      Homeostatic
  Qwen       r=256 87.8      83.1%      78.0%      Degradative
  Qwen Full  r=4   3.4       94.9%      -          Homeostatic
  Qwen Full  r=16  11.5      91.1%      87.3%      Quasi-homeostatic
  -----------------------------------------------------------------
  Gemma      r=2   1.75      95.7%      -          Homeostatic
  Gemma      r=4   3.1       91.5%      93.6%      Homeostatic
  Gemma      r=16  9.2       70.2%      -          Degradative
  Gemma      r=256 71.5      70.2%      -          Degradative
  -----------------------------------------------------------------

  THRESHOLD COMPARISON:
    Qwen:  transition between eff_rank ~50 (r=128) and ~88 (r=256)
    Gemma: transition between eff_rank ~3 (r=4) and ~9 (r=16)

  The regime transition is ARCHITECTURE-GENERAL but THRESHOLD-DEPENDENT.
""")
