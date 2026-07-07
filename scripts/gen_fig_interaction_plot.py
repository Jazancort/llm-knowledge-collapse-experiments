"""
Figure: Rank × LR Interaction Plot (replaces heatmap)
Shows 3 lines (one per rank) with LR on x-axis and Gen5 retention on y-axis.
Non-parallel lines demonstrate interaction between rank and LR.

Matches visual style of fig1_trajectories (Wong palette, Arial, frameon=False legend).
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

FIG_DIR = Path(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v2\figs")
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Data: Rank × LR interaction matrix (Qwen 2.5 1.5B, K0=78, Gen5, seed 15)
lrs = [5e-6, 1e-5, 2e-5]
lr_labels = [r'$5 \times 10^{-6}$', r'$1 \times 10^{-5}$', r'$2 \times 10^{-5}$']

# Retention counts out of K0=78
retention_counts = {
    16:  [76, 76, 72],
    64:  [76, 72, 69],
    256: [70, 66, 64],
}
K0 = 78

# Convert to percentages
retention_pct = {r: [c / K0 * 100 for c in counts] for r, counts in retention_counts.items()}

# Colors from Wong palette (same as fig1_trajectories)
colors = {
    16:  '#0072B2',   # blue
    64:  '#E69F00',   # orange
    256: '#D55E00',   # vermillion
}
markers = {
    16:  'o',
    64:  's',
    256: '^',
}

# Plot
fig, ax = plt.subplots(figsize=(3.5, 2.6))

# Regime bands (same as fig1_trajectories for visual consistency)
ax.axhspan(90, 102, color='#D1FAE5', alpha=0.4, zorder=0)
ax.axhspan(80, 90, color='#FDE68A', alpha=0.3, zorder=0)
ax.axhspan(60, 80, color='#FECACA', alpha=0.3, zorder=0)

# Threshold lines (subtle — regime bands already do the heavy lifting)
ax.axhline(90, color='#065F46', linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)
ax.axhline(80, color='#991B1B', linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)

# Plot lines
x_pos = [0, 1, 2]  # categorical x positions

for rank in [16, 64, 256]:
    ax.plot(x_pos, retention_pct[rank],
            color=colors[rank], linewidth=2.0, marker=markers[rank],
            markersize=7, markeredgecolor='white', markeredgewidth=0.8,
            label=f'$r = {rank}$', zorder=3)

# Annotate values with manual offsets to avoid overlap
# Format: (rank, point_index): (x_offset, y_offset)
offsets = {
    (16, 0): (12, 2),    # r=16 at LR=5e-6: right (same value as r=64)
    (16, 1): (0, 6),     # r=16 at LR=1e-5: above
    (16, 2): (12, 0),    # r=16 at LR=2e-5: right (avoid r=64)
    (64, 0): (0, 6),     # r=64 at LR=5e-6: above
    (64, 1): (12, 0),    # r=64 at LR=1e-5: right
    (64, 2): (12, 0),    # r=64 at LR=2e-5: right
    (256, 0): (0, -9),   # r=256 at LR=5e-6: below
    (256, 1): (0, -9),   # r=256 at LR=1e-5: below
    (256, 2): (0, -9),   # r=256 at LR=2e-5: below
}

for rank in [16, 64, 256]:
    for i, val in enumerate(retention_pct[rank]):
        # Skip r=16 labels where value matches r=64 (avoid clutter)
        if rank == 16 and i < 2 and abs(val - retention_pct[64][i]) < 0.1:
            continue
        ox, oy = offsets[(rank, i)]
        ha = 'left' if ox > 0 else 'center'
        ax.annotate(f'{val:.1f}%', (x_pos[i], val),
                    textcoords='offset points', xytext=(ox, oy),
                    ha=ha, fontsize=7, color=colors[rank], fontweight='bold')

# Regime labels on right
ax.text(2.35, 96, 'Homeostatic', fontsize=7, color='#065F46', va='center')
ax.text(2.35, 85, 'Bounded', fontsize=7, color='#78350F', va='center')
ax.text(2.35, 78, 'Degradative', fontsize=7, color='#991B1B', va='center')

# Axes
ax.set_xticks(x_pos)
ax.set_xticklabels(lr_labels, fontsize=9)
ax.set_xlabel('Learning Rate', fontsize=9)
ax.set_ylabel('$K_0$ Retention at Gen5 (%)', fontsize=9)
ax.set_xlim(-0.3, 2.5)
ax.set_ylim(76, 102)

# Grid and spines (same style)
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend below (same style as fig1) with title
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=False, fontsize=7.5, title='Adapter rank', title_fontsize=8)

# Adjust for regime labels
plt.subplots_adjust(right=0.82)

# Save
fig.savefig(FIG_DIR / "fig_rank_lr_heatmap.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'fig_rank_lr_heatmap.png'}")
plt.close()
