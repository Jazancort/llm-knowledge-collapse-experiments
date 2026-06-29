"""
Figure 3: Cross-Backbone Comparison
Retention vs Effective Rank, different shapes per backbone, colors per regime.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from pathlib import Path

plt.style.use(["science", "ieee", "no-latex"])
FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

# Data: (effective_rank, retention_%, backbone, regime)
# Qwen 2.5 1.5B (Gen10 where available, Gen5 for r=4/32)
qwen_data = [
    (3.34,  96.2, "homeostatic"),    # r=4
    (11.08, 94.9, "homeostatic"),    # r=16
    (17.85, 94.9, "homeostatic"),    # r=32
    (29.52, 91.1, "homeostatic"),    # r=64
    (50.16, 88.6, "transition"),     # r=128
    (87.57, 78.0, "degradative"),    # r=256
]

# Gemma 3 1B (Gen5 mean for N=3)
gemma3_data = [
    (1.75,  97.9, "homeostatic"),    # r=2
    (3.07,  92.9, "homeostatic"),    # r=4 mean(93.6, 93.6, 91.5)
    (9.17,  68.8, "degradative"),    # r=16 mean(70.2, 68.1, 68.1)
    (71.47, 70.2, "degradative"),    # r=256
]

# Gemma 4 E2B (Gen5, N=1)
gemma4_data = [
    (2.13,  96.1, "homeostatic"),    # r=4
    (5.64,  97.4, "homeostatic"),    # r=16
]

# Color map
regime_colors = {
    "homeostatic": "C0",
    "transition": "C1",
    "degradative": "C3",
}

fig, ax = plt.subplots(figsize=(3.3, 2.4))

# Plot each backbone with different marker
for er, ret, regime in qwen_data:
    ax.scatter(er, ret, marker="o", s=50, color=regime_colors[regime],
               edgecolors="black", linewidth=0.5, zorder=4)

for er, ret, regime in gemma3_data:
    ax.scatter(er, ret, marker="s", s=50, color=regime_colors[regime],
               edgecolors="black", linewidth=0.5, zorder=4)

for er, ret, regime in gemma4_data:
    ax.scatter(er, ret, marker="^", s=55, color=regime_colors[regime],
               edgecolors="black", linewidth=0.5, zorder=4)

# Connect points within same backbone (guide lines)
qwen_er = [d[0] for d in qwen_data]
qwen_ret = [d[1] for d in qwen_data]
ax.plot(qwen_er, qwen_ret, color="gray", alpha=0.4, linewidth=0.8, zorder=2)

gemma3_er = [d[0] for d in gemma3_data]
gemma3_ret = [d[1] for d in gemma3_data]
ax.plot(gemma3_er, gemma3_ret, color="gray", alpha=0.4, linewidth=0.8,
        linestyle="--", zorder=2)

# Legend: backbone shapes
from matplotlib.lines import Line2D
legend_backbone = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=7, markeredgecolor="black", markeredgewidth=0.5,
           label="Qwen 2.5 1.5B"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
           markersize=7, markeredgecolor="black", markeredgewidth=0.5,
           label="Gemma 3 1B"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
           markersize=7, markeredgecolor="black", markeredgewidth=0.5,
           label="Gemma 4 E2B"),
]
legend_regime = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="C0",
           markersize=7, label="Homeostatic"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="C1",
           markersize=7, label="Transition"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="C3",
           markersize=7, label="Degradative"),
]

leg1 = ax.legend(handles=legend_backbone, loc="upper right", fontsize=6,
                 title="Backbone", title_fontsize=6, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=legend_regime, loc="lower left", fontsize=6,
          title="Regime", title_fontsize=6, framealpha=0.9)

# Formatting
ax.set_xlabel("Mean Effective Rank")
ax.set_ylabel("K0 Retention (%)")
ax.set_xscale("log")
ax.set_xlim(1, 120)
ax.set_ylim(60, 102)
ax.grid(True, alpha=0.3)

# Save
fig.savefig(FIG_DIR / "fig3_cross_backbone.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig3_cross_backbone.svg", bbox_inches="tight")
print("Saved fig3_cross_backbone.pdf and .svg")
plt.close()
