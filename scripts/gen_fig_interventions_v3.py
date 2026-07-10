"""Figure 8: Intervention comparison on Qwen r=256 at Gen5.
Now includes C4 and error bars (range) for C3/C5.
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

K0 = 78

# Data from intervention experiments
conditions = ['C1\nNormal', 'C2\nShort', 'C3\nFiltered', 'C4\nCanonical', 'C5\nRandom']
retention_pct = [83.3, 91.0, 92.3, 88.5, 92.7]

# Error bars (range) — only C3 and C5 have replicates
# C3 seeds: 73/78, 71/78, 72/78
c3_vals = np.array([73, 71, 72]) / K0 * 100
# C5 masks: 72/78, 73/78, 72/78
c5_vals = np.array([72, 73, 72]) / K0 * 100

# Errors as (lower, upper) distance from mean
errors_low = [0, 0, retention_pct[2] - c3_vals.min(), 0, retention_pct[4] - c5_vals.min()]
errors_high = [0, 0, c3_vals.max() - retention_pct[2], 0, c5_vals.max() - retention_pct[4]]

# Colors
colors = ['#888888', '#56B4E9', '#009E73', '#CC79A7', '#E69F00']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.8), gridspec_kw={'height_ratios': [3, 2]})

# --- Panel (a): Retention by condition ---
bars = ax1.bar(range(5), retention_pct, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax1.errorbar(range(5), retention_pct,
             yerr=[errors_low, errors_high],
             fmt='none', ecolor='#333333', elinewidth=1.2, capsize=4, capthick=1, zorder=4)

# Baseline line
ax1.axhline(83.3, color='#888888', linewidth=0.8, linestyle=':', alpha=0.7, zorder=2)

# Regime bands (subtle)
ax1.axhspan(90, 100, color='#D1FAE5', alpha=0.2, zorder=0)
ax1.axhline(90, color='#065F46', linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)

ax1.set_xticks(range(5))
ax1.set_xticklabels(conditions, fontsize=7.5)
ax1.set_ylabel('$K_0$ Retention at Gen5 (%)', fontsize=8)
ax1.set_ylim(78, 97)
ax1.set_title('(a) Retention by condition', fontsize=9, fontweight='bold', loc='left')
ax1.grid(True, axis='y', alpha=0.12, linewidth=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Value labels on bars — position above error bar end
for i, v in enumerate(retention_pct):
    err_top = v + errors_high[i]
    ax1.text(i, err_top + 0.5, f'{v:.1f}%', ha='center', fontsize=7, fontweight='bold', color=colors[i])

# --- Panel (b): Improvement over C1 ---
delta = [r - retention_pct[0] for r in retention_pct[1:]]
delta_colors = colors[1:]
delta_labels = ['C2', 'C3', 'C4', 'C5']
delta_errors_low = errors_low[1:]
delta_errors_high = errors_high[1:]

bars2 = ax2.barh(range(4), delta, color=delta_colors, edgecolor='white', linewidth=0.5, zorder=3)
ax2.errorbar(delta, range(4),
             xerr=[delta_errors_low, delta_errors_high],
             fmt='none', ecolor='#333333', elinewidth=1.2, capsize=3, capthick=1, zorder=4)

ax2.set_yticks(range(4))
ax2.set_yticklabels(delta_labels, fontsize=8)
ax2.set_xlabel('$\\Delta$ vs C1 (pp)', fontsize=8)
ax2.set_title('(b) Improvement over baseline', fontsize=9, fontweight='bold', loc='left')
ax2.set_xlim(0, 12)
ax2.grid(True, axis='x', alpha=0.12, linewidth=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Value labels — position after error bar end
for i, v in enumerate(delta):
    err_end = v + delta_errors_high[i]
    ax2.text(err_end + 0.3, i, f'+{v:.1f}pp', va='center', fontsize=7, fontweight='bold', color=delta_colors[i])

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_interventions.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'fig_interventions.png'}")
plt.close()
