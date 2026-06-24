"""Generate the 4 main paper figures from experimental data."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

# === DATA ===

gens = np.arange(0, 11)

# r=16 (seed 15, 10 gens)
r16 = [100, 94.9, 94.9, 94.9, 94.9, 94.9, 96.2, 94.9, 96.2, 94.9, 94.9]

# r=128 (seed 15, 10 gens)
r128 = [100, 92.4, 88.6, 91.1, 91.1, 87.3, 87.3, 88.6, 87.3, 87.3, 88.6]

# r=256 (3 seeds, 10 gens)
r256_s15 =  [100, 94.9, 89.9, 91.1, 88.6, 83.5, 82.3, 81.0, 81.0, 78.5, 75.9]
r256_s137 = [100, 94.9, 89.9, 87.3, 86.1, 84.8, 84.8, 82.3, 81.0, 78.5, 77.2]
r256_s256 = [100, 94.9, 89.9, 89.9, 89.9, 81.0, 83.5, 79.7, 81.0, 77.2, 81.0]

r256_mean = np.mean([r256_s15, r256_s137, r256_s256], axis=0)
r256_std = np.std([r256_s15, r256_s137, r256_s256], axis=0)

# Dose-response (Gen 5 values)
ranks_nominal = [4, 16, 64, 128, 256]
retention_gen5 = [96.2, 94.9, 89.9, 87.3, 83.1]
eff_ranks = [3.34, 11.08, 30.08, 50.5, 87.8]
utilization = [84, 69, 47, 39, 34]

# Transitions r=256 seed 15
cw_256 = [0, 0, 4, 3, 5, 7, 4, 2, 1, 3, 3]  # C->W per gen
wc_256 = [0, 0, 1, 7, 1, 4, 3, 1, 0, 1, 0]  # W->C per gen

# Transitions r=128 seed 15
cw_128 = [0, 0, 3, 0, 2, 1, 0, 0, 3, 2, 1]
wc_128 = [0, 0, 0, 3, 2, 0, 0, 2, 2, 3, 4]


# === FIGURE 1: Longitudinal Trajectories ===

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(gens, r16, 'o-', color='#2196F3', linewidth=2, markersize=6, label='r = 16')
ax.plot(gens, r128, 's-', color='#FF9800', linewidth=2, markersize=6, label='r = 128')
ax.plot(gens, r256_mean, 'D-', color='#E53935', linewidth=2, markersize=6, label='r = 256 (mean ± std)')
ax.fill_between(gens, r256_mean - r256_std, r256_mean + r256_std, color='#E53935', alpha=0.15)

ax.set_xlabel('Generation')
ax.set_ylabel('K₀ Retention (%)')
ax.set_title('Factual Retention Under Recursive Synthetic Fine-Tuning')
ax.set_xlim(0, 10)
ax.set_ylim(70, 102)
ax.set_xticks(gens)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.axhline(y=78, color='#E53935', linestyle=':', alpha=0.4)
ax.axhline(y=94.9, color='#2196F3', linestyle=':', alpha=0.4)

plt.savefig(OUTPUT_DIR / "fig1_longitudinal_trajectories.png")
plt.savefig(OUTPUT_DIR / "fig1_longitudinal_trajectories.pdf")
plt.close()
print("Fig 1: Longitudinal trajectories - DONE")


# === FIGURE 2: Dose-Response (dual axis) ===

fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = '#E53935'
ax1.set_xlabel('Nominal Rank')
ax1.set_ylabel('Factual Loss at Gen 5 (%)', color=color1)
loss = [100 - r for r in retention_gen5]
ax1.plot(ranks_nominal, loss, 'D-', color=color1, linewidth=2, markersize=8, label='Factual Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 25)

ax2 = ax1.twinx()
color2 = '#1565C0'
ax2.set_ylabel('Effective Rank', color=color2)
ax2.plot(ranks_nominal, eff_ranks, 's-', color=color2, linewidth=2, markersize=8, label='Effective Rank')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 100)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

ax1.set_title('Dose-Response: Adapter Rank vs Factual Degradation')
ax1.set_xscale('log', base=2)
ax1.set_xticks(ranks_nominal)
ax1.set_xticklabels([str(r) for r in ranks_nominal])
ax1.grid(True, alpha=0.3)

plt.savefig(OUTPUT_DIR / "fig2_dose_response.png")
plt.savefig(OUTPUT_DIR / "fig2_dose_response.pdf")
plt.close()
print("Fig 2: Dose-response curve - DONE")


# === FIGURE 3: Plasticity Depletion (Transitions) ===

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# r=128
x = gens[2:]  # transitions start at gen 2
ax1.bar(x - 0.15, cw_128[2:], 0.3, color='#E53935', alpha=0.8, label='C → W (lost)')
ax1.bar(x + 0.15, wc_128[2:], 0.3, color='#4CAF50', alpha=0.8, label='W → C (recovered)')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Factual Transitions per Generation')
ax1.set_title('r = 128 (Homeostatic)')
ax1.legend()
ax1.set_xticks(x)
ax1.set_ylim(0, 10)
ax1.grid(True, alpha=0.3, axis='y')

# r=256
ax2.bar(x - 0.15, cw_256[2:], 0.3, color='#E53935', alpha=0.8, label='C → W (lost)')
ax2.bar(x + 0.15, wc_256[2:], 0.3, color='#4CAF50', alpha=0.8, label='W → C (recovered)')
ax2.set_xlabel('Generation')
ax2.set_title('r = 256 (Degradative)')
ax2.legend()
ax2.set_xticks(x)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Factual Transition Dynamics: Plasticity Depletion', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_plasticity_depletion.png")
plt.savefig(OUTPUT_DIR / "fig3_plasticity_depletion.pdf")
plt.close()
print("Fig 3: Plasticity depletion - DONE")


# === FIGURE 4: Utilization Curve ===

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(ranks_nominal, utilization, 'o-', color='#6A1B9A', linewidth=2.5, markersize=9)
for i, (r, u) in enumerate(zip(ranks_nominal, utilization)):
    ax.annotate(f'{u}%', (r, u), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10)

ax.set_xlabel('Nominal Rank')
ax.set_ylabel('Effective Rank Utilization (%)')
ax.set_title('Diminishing Returns: Adapter Capacity vs Actual Utilization')
ax.set_xscale('log', base=2)
ax.set_xticks(ranks_nominal)
ax.set_xticklabels([str(r) for r in ranks_nominal])
ax.set_ylim(20, 100)
ax.grid(True, alpha=0.3)

# Trend annotation
ax.annotate('84% → 34%\n(rank ×64)', xy=(64, 55), fontsize=9, color='#6A1B9A', alpha=0.7, ha='center')

plt.savefig(OUTPUT_DIR / "fig4_utilization_curve.png")
plt.savefig(OUTPUT_DIR / "fig4_utilization_curve.pdf")
plt.close()
print("Fig 4: Utilization curve - DONE")

print(f"\nAll figures saved to: {OUTPUT_DIR}")
