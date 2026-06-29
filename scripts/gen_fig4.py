"""
Figure 4: FFT vs QLoRA Gen10 Trajectories (3 seeds each)
Shows one-time adaptation gap at Gen1, both flat afterward.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from pathlib import Path

plt.style.use(["science", "ieee", "no-latex"])

OUT = Path(__file__).parent.parent / "outputs" / "fft_drift_gen10"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

K0 = 78

def load(method, seed):
    data = json.loads((OUT / f"{method}_seed{seed}.json").read_text())
    gens = [0] + [r["gen"] for r in data]
    ret = [100.0] + [r["retention"] / K0 * 100 for r in data]
    return gens, ret

fig, ax = plt.subplots(figsize=(3.3, 1.8))

# QLoRA r=16 (blue)
qlora_all = []
for seed in [15, 137, 256]:
    g, r = load("qlora_r16", seed)
    qlora_all.append(r)
    ax.plot(g, r, color="C0", alpha=0.35, linewidth=0.7, zorder=2)
qlora_mean = np.mean(qlora_all, axis=0)
ax.plot(list(range(11)), qlora_mean, color="C0", linewidth=1.8,
        label="QLoRA $r=16$ (N=3)", zorder=3)

# FFT LR=1e-6 (red)
fft_all = []
for seed in [15, 137, 256]:
    g, r = load("fft_lr1e-06", seed)
    fft_all.append(r)
    ax.plot(g, r, color="C3", alpha=0.35, linewidth=0.7, zorder=2)
fft_mean = np.mean(fft_all, axis=0)
ax.plot(list(range(11)), fft_mean, color="C3", linewidth=1.8,
        linestyle="--", label="FFT LR=$10^{-6}$ (N=3)", zorder=3)

# Annotation: one-time gap
gap_gen1 = qlora_mean[1] - fft_mean[1]
mid_y = (qlora_mean[1] + fft_mean[1]) / 2
ax.annotate("", xy=(1, fft_mean[1]), xytext=(1, qlora_mean[1]),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
ax.text(1.4, mid_y, f"$\\approx${gap_gen1:.1f}pp", fontsize=6.5,
        va="center", ha="left")

# Formatting
ax.set_xlabel("Generation")
ax.set_ylabel("K0 Retention (%)")
ax.set_xlim(0, 10)
ax.set_ylim(85, 102)
ax.set_xticks(range(0, 11, 2))
ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
ax.grid(True, alpha=0.3)

# Save
fig.savefig(FIG_DIR / "fig4_fft_vs_qlora.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig4_fft_vs_qlora.svg", bbox_inches="tight")
print("Saved fig4_fft_vs_qlora.pdf and .svg")
plt.close()
