"""Replot Fig 4 — FFT vs QLoRA with mean ± std shaded bands."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.5,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
})

K0 = 78
generations = np.arange(0, 11)

# Raw retention counts (gen 1-10), prepend K0 for gen 0
fft_seeds = {
    15:  [K0, 72, 72, 71, 72, 71, 72, 72, 72, 71, 72],
    137: [K0, 72, 73, 72, 72, 72, 72, 72, 72, 71, 71],
    256: [K0, 72, 71, 72, 72, 72, 72, 71, 72, 72, 72],
}
qlora_seeds = {
    15:  [K0, 76, 76, 76, 76, 75, 76, 76, 75, 76, 76],
    137: [K0, 75, 76, 76, 76, 76, 75, 76, 76, 76, 76],
    256: [K0, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76],
}

# Convert to percentages
fft_pct = np.array([list(v) for v in fft_seeds.values()]) / K0 * 100
qlora_pct = np.array([list(v) for v in qlora_seeds.values()]) / K0 * 100

fft_mean = fft_pct.mean(axis=0)
fft_std = fft_pct.std(axis=0)
qlora_mean = qlora_pct.mean(axis=0)
qlora_std = qlora_pct.std(axis=0)

# Colors (Wong palette)
c_qlora = "#0072B2"  # blue
c_fft = "#E69F00"    # orange

fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=300)

# Shaded bands
ax.fill_between(generations, qlora_mean - qlora_std, qlora_mean + qlora_std,
                alpha=0.2, color=c_qlora, linewidth=0)
ax.fill_between(generations, fft_mean - fft_std, fft_mean + fft_std,
                alpha=0.2, color=c_fft, linewidth=0)

# Mean lines
ax.plot(generations, qlora_mean, color=c_qlora, marker="o", markersize=3.5,
        label="QLoRA $r{=}16$ (N=3)", zorder=3)
ax.plot(generations, fft_mean, color=c_fft, marker="s", markersize=3.5,
        label="FFT, LR $= 10^{-6}$ (N=3)", zorder=3)

ax.set_xlabel("Generation")
ax.set_ylabel("$K_0$ Retention (%)")
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(85, 101)
ax.set_xticks(range(0, 11, 2))
ax.set_yticks([85, 90, 95, 100])
ax.grid(True, axis="y", linestyle="--", alpha=0.4)

# Legend below the plot (same style as other figures)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
          fontsize=7.5, frameon=False, columnspacing=1.5)

plt.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.22)
out_path = r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v3\figs\fig4_fft_vs_qlora"
plt.savefig(out_path + ".pdf", bbox_inches="tight")
plt.savefig(out_path + ".png", bbox_inches="tight", dpi=300)
plt.savefig(out_path + ".svg", bbox_inches="tight")
print("Saved fig4_fft_vs_qlora (.pdf, .png, .svg)")
print(f"QLoRA mean: {qlora_mean}")
print(f"QLoRA std:  {qlora_std}")
print(f"FFT mean:   {fft_mean}")
print(f"FFT std:    {fft_std}")
