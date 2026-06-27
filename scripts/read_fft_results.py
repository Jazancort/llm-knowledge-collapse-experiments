import json

print("FFT vs QLoRA — CAUSAL CONTROL")
print("="*60)

for method in ["fft", "qlora"]:
    data = json.load(open(f"outputs/fft_comparison_{method}_seed15/results.json"))
    k0 = data[0].get("k0_size", 79)
    print(f"\n  {method.upper()} (K0={k0}):")
    for r in data:
        gen = r["generation"]
        ret = r["retention"]
        pct = ret / k0 * 100
        mi = r.get("method_info")
        trans = r.get("transitions")
        cw = str(trans["C->W"]) if trans else "-"
        wc = str(trans["W->C"]) if trans else "-"
        extra = ""
        if mi:
            if "eff_rank" in mi:
                extra = "eff_rank=" + str(round(mi["eff_rank"], 2))
            elif "weight_drift" in mi:
                extra = "drift=" + str(round(mi["weight_drift"]["relative_drift"], 6))
        print(f"    Gen {gen}: {ret}/{k0} ({pct:.1f}%) {extra} C->W={cw} W->C={wc}")
