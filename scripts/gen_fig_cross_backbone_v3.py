"""Figure: Cross-backbone retention vs effective rank (Qwen + Gemma 3 + E2B).
Shows regime transition at different effective rank values per backbone.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

FIG_DIR = Path(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v3\figs")

# Data points: (effective_rank, retention_pct, regime)
# Qwen (Gen10, from existing experiments)
qwen = [
    (3.34, 97.4, 'homeo'),    # r=4
    (11.08, 96.2, 'homeo'),   # r=16
    (17.85, 96.2, 'homeo'),   # r=32
    (29.52, 92.3, 'homeo'),   # r=64
    (50.16, 89.7, 'bounded'), # r=128
    (87.57, 79.1, 'degrad'),  # r=256
]

# Gemma 3 (Gen10, from v3 experiments)
gemma3 = [
    (3.07, 94.4, 'homeo'),    # r=4 (N=5 mean)
    (6.37, 78.3, 'degrad'),   # r=10
    (7.35, 73.9, 'degrad'),   # r=12
    (8.37, 69.6, 'degrad'),   # r=14
    (9.17, 56.5, 'degrad'),   # r=16 (N=5 mean, Gen10)
]

# E2B (Gen10, from v3 experiments)
e2b = [
    (2.04, 94.7, 'homeo'),    # r=4
    (5.10, 88.2, 'bounded'),  # r=16
    (13.28, 75.0, 'degrad'),  # r=64
]

# Colors by regime
regime_colors = {'homeo': '#0072B2', 'bounded': '#E69F00', 'degrad': '#D55E00'}

fig, ax = plt.subplots(figsize=(3.5, 2.6))

# Regime bands
ax.axhspan(90, 102, color='#D1FAE5', alpha=0.35, zorder=0, clip_on=True)
ax.axhspan(80, 90, color='#FDE68A', alpha=0.25, zorder=0, clip_on=True)
ax.axhspan(45, 80, color='#FECACA', alpha=0.25, zorder=0, clip_on=True)
ax.axhline(90, color='#065F46', linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)
ax.axhline(80, color='#991B1B', linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)

# Plot each backbone
for er, ret, regime in qwen:
    ax.scatter(er, ret, color=regime_colors[regime], marker='o', s=50,
               edgecolors='white', linewidths=0.5, zorder=4)
for er, ret, regime in gemma3:
    ax.scatter(er, ret, color=regime_colors[regime], marker='s', s=50,
               edgecolors='white', linewidths=0.5, zorder=4)
for er, ret, regime in e2b:
    ax.scatter(er, ret, color=regime_colors[regime], marker='^', s=60,
               edgecolors='white', linewidths=0.5, zorder=4)

# Legend entries (manual for backbone shapes)
ax.scatter([], [], color='#555', marker='o', s=40, label='Qwen 2.5 1.5B')
ax.scatter([], [], color='#555', marker='s', s=40, label='Gemma 3 1B')
ax.scatter([], [], color='#555', marker='^', s=50, label='Gemma 4 E2B')

# Axes
ax.set_xscale('log')
ax.set_xlim(1.5, 120)
ax.set_ylim(45, 102)
ax.set_xlabel('Mean Effective Rank', fontsize=9)
ax.set_ylabel('$K_0$ Retention at Gen10 (%)', fontsize=9)
ax.grid(True, alpha=0.12, linewidth=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Regime labels
ax.text(110, 96, 'Homeo.', fontsize=6.5, color='#065F46', va='center')
ax.text(110, 85, 'Bounded', fontsize=6.5, color='#78350F', va='center')
ax.text(110, 62, 'Degrad.', fontsize=6.5, color='#991B1B', va='center')

# Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=False, fontsize=7.5)

plt.subplots_adjust(right=0.85)
fig.savefig(FIG_DIR / "fig_cross_backbone.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'fig_cross_backbone.png'}")
plt.close()
