"""Generate Rank × LR heatmap figure using matplotlib.
Green = high retention (good), Red = low retention (bad).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.linewidth': 0.8,
})

FIG_DIR = Path(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v2\figs")
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Data from rank_lr_matrix experiment (Qwen 2.5 1.5B, K0=78, Gen5, seed 15)
ranks = [16, 64, 256]
lrs = [r"$5{\times}10^{-6}$", r"$10^{-5}$", r"$2{\times}10^{-5}$"]
retention_raw = np.array([
    [76, 76, 72],   # r=16
    [76, 72, 69],   # r=64
    [70, 66, 64],   # r=256
])
retention_pct = retention_raw / 78 * 100

# Labels for cells
labels = np.array([
    ["97.4%", "97.4%", "92.3%"],
    ["97.4%", "92.3%", "88.5%"],
    ["89.7%", "84.6%", "82.1%"],
])

# Custom colormap: green (high/good) → yellow (mid) → red (low/bad)
colors = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
cmap = LinearSegmentedColormap.from_list("regime", colors, N=256)

fig, ax = plt.subplots(figsize=(3.5, 2.4))

im = ax.imshow(retention_pct, cmap=cmap, vmin=80, vmax=100, aspect='auto')

# Add text annotations
for i in range(3):
    for j in range(3):
        val = retention_pct[i, j]
        # Dark text for light backgrounds, white for dark
        color = 'white' if val < 85 else '#1a1a1a'
        ax.text(j, i, labels[i, j], ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)

# Axes
ax.set_xticks(range(3))
ax.set_xticklabels(lrs, fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels([f"$r = {r}$" for r in ranks], fontsize=9)
ax.set_xlabel("Learning Rate", fontsize=9)
ax.set_ylabel("Adapter Rank", fontsize=9)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Gen5 Retention (%)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_rank_lr_heatmap.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'fig_rank_lr_heatmap.png'}")
plt.close()
