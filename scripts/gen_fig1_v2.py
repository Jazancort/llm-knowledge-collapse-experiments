"""
Figure: Longitudinal K0 Retention Trajectories with regime bands
Qwen 2.5 1.5B, r=16 (N=3), r=128 (N=1), r=256 (N=3), Gen0-10
"""
import json
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

OUT = Path(__file__).parent.parent / "outputs"
FIG_DIR = Path(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v2\figs")
FIG_DIR.mkdir(exist_ok=True, parents=True)


def load_g1_retention(folder):
    data = json.loads((OUT / folder / "results.json").read_text())
    k0 = data[0]["k0_size"]
    gens = [r["generation"] for r in data]
    ret = [r["retention"] / k0 * 100 for r in data]
    return gens, ret


def load_fft_gen10(filename, k0=78):
    data = json.loads((OUT / "fft_drift_gen10" / filename).read_text())
    gens = [0] + [r["gen"] for r in data]
    ret = [100.0] + [r["retention"] / k0 * 100 for r in data]
    return gens, ret


# Load data
r16_seeds = {}
for seed in [15, 137, 256]:
    r16_seeds[seed] = load_fft_gen10(f"qlora_r16_seed{seed}.json")

r128_gens, r128_ret = load_g1_retention("g1_rank128_seed15")

r256_seeds = {}
for seed in [15, 137, 256]:
    r256_seeds[seed] = load_g1_retention(f"g1_rank256_seed{seed}")

# Plot
fig, ax = plt.subplots(figsize=(3.5, 2.6))

# Regime bands (background)
ax.axhspan(90, 102, color='#D1FAE5', alpha=0.4, zorder=0, label='_nolegend_')
ax.axhspan(80, 90, color='#FDE68A', alpha=0.3, zorder=0, label='_nolegend_')
ax.axhspan(60, 80, color='#FECACA', alpha=0.3, zorder=0, label='_nolegend_')

# Threshold lines at regime boundaries
ax.axhline(90, color='#065F46', linewidth=0.6, linestyle='--', alpha=0.6, zorder=1)
ax.axhline(80, color='#991B1B', linewidth=0.6, linestyle='--', alpha=0.6, zorder=1)

# Regime labels on right
ax.text(10.3, 96, 'Homeostatic', fontsize=7, color='#065F46', va='center')
ax.text(10.3, 85, 'Bounded', fontsize=7, color='#78350F', va='center')
ax.text(10.3, 70, 'Degradative', fontsize=7, color='#991B1B', va='center')

# r=16 (blue) - per-seed lines + mean
for seed, (g, r) in r16_seeds.items():
    ax.plot(g, r, color='#0072B2', alpha=0.25, linewidth=0.7, zorder=2)
r16_mean = np.mean([r16_seeds[s][1] for s in r16_seeds], axis=0)
ax.plot(r16_seeds[15][0], r16_mean, color='#0072B2', linewidth=2.0,
        label='$r=16$ (N=3)', zorder=3)

# r=128 (orange) - single seed
ax.plot(r128_gens, r128_ret, color='#E69F00', linewidth=2.0, linestyle='--',
        label='$r=128$ (N=1)', zorder=3)

# r=256 (vermillion) - per-seed lines + mean
for seed, (g, r) in r256_seeds.items():
    ax.plot(g, r, color='#D55E00', alpha=0.25, linewidth=0.7, zorder=2)
r256_mean = np.mean([r256_seeds[s][1] for s in r256_seeds], axis=0)
ax.plot(r256_seeds[15][0], r256_mean, color='#D55E00', linewidth=2.0,
        label='$r=256$ (N=3)', zorder=3)

# Formatting
ax.set_xlabel('Generation', fontsize=9)
ax.set_ylabel('$K_0$ Retention (%)', fontsize=9)
ax.set_xlim(0, 10)
ax.set_ylim(60, 102)
ax.set_xticks(range(0, 11, 2))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=False, fontsize=7.5)
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust for regime labels on right
plt.subplots_adjust(right=0.82)

# Save
fig.savefig(FIG_DIR / "fig1_trajectories.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'fig1_trajectories.png'}")
plt.close()
