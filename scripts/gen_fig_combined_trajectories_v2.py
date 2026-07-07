"""Figure: Combined Qwen + Gemma 3 trajectories (matplotlib, stacked).
Addresses Carlos #35-37: regime bands, no confusing dotted lines, clear legend.
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

# Colors (Wong palette)
C_HOMEO = '#0072B2'
C_BOUNDED = '#E69F00'
C_DEG = '#D55E00'
C_INT = '#CC79A7'


def load_fft_gen10(filename, k0=78):
    data = json.loads((OUT / "fft_drift_gen10" / filename).read_text())
    gens = [0] + [r["gen"] for r in data]
    ret = [100.0] + [r["retention"] / k0 * 100 for r in data]
    return gens, ret


def load_g1_retention(folder, k0_override=None):
    data = json.loads((OUT / folder / "results.json").read_text())
    k0 = k0_override or data[0]["k0_size"]
    gens = [r["generation"] for r in data]
    ret = [r["retention"] / k0 * 100 for r in data]
    return gens, ret


def load_gemma3(path, k0=47):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and 'generations' in data:
        k0 = data.get('k0_size', k0)
        gens = [g['gen'] for g in data['generations']]
        ret = [g['retention'] / k0 * 100 for g in data['generations']]
    elif isinstance(data, list) and data and 'generation' in data[0]:
        k0 = data[0].get('k0_size', k0)
        gens = [g['generation'] for g in data if g.get('generation', 0) > 0]
        ret = [g['retention'] / k0 * 100 for g in data if g.get('generation', 0) > 0]
    else:
        return [], []
    return gens, ret


# === Load Qwen data ===
qwen_r16 = [load_fft_gen10(f"qlora_r16_seed{s}.json") for s in [15, 137, 256]]
qwen_r128_g, qwen_r128_r = load_g1_retention("g1_rank128_seed15")
qwen_r256 = [load_g1_retention(f"g1_rank256_seed{s}") for s in [15, 137, 256]]

# === Load Gemma 3 data ===
gemma_r4 = []
for seed in [15, 137, 256]:
    path = OUT / f'gemma3_rank4_seed{seed}' / 'results.json'
    if path.exists():
        gemma_r4.append(load_gemma3(path))
for seed in [42, 77]:
    path = OUT / 'gemma3_extra_seeds' / f'gemma3_rank4_seed{seed}.json'
    if path.exists():
        gemma_r4.append(load_gemma3(path))

gemma_r16 = []
for seed in [15, 137, 256]:
    path = OUT / f'gemma3_rank16_seed{seed}' / 'results.json'
    if path.exists():
        gemma_r16.append(load_gemma3(path))
for seed in [42, 77]:
    path = OUT / 'gemma3_extra_seeds' / f'gemma3_rank16_seed{seed}.json'
    if path.exists():
        gemma_r16.append(load_gemma3(path))


def compute_band(trajectories):
    """Compute mean, min, max across seeds."""
    min_len = min(len(t[1]) for t in trajectories)
    arr = np.array([t[1][:min_len] for t in trajectories])
    gens = trajectories[0][0][:min_len]
    return gens, arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)


def add_regime_bands(ax, ymin=55):
    """Add colored regime background bands."""
    ax.axhspan(90, 105, color='#D1FAE5', alpha=0.35, zorder=0)
    ax.axhspan(80, 90, color='#FDE68A', alpha=0.25, zorder=0)
    ax.axhspan(ymin, 80, color='#FECACA', alpha=0.25, zorder=0)
    ax.axhline(90, color='#065F46', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(80, color='#991B1B', linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)


# === PLOT ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=False)

# --- Panel (a): Qwen ---
add_regime_bands(ax1, ymin=60)

# r=16 band
g16, mean16, min16, max16 = compute_band(qwen_r16)
ax1.fill_between(g16, min16, max16, color=C_HOMEO, alpha=0.15, zorder=2)
ax1.plot(g16, mean16, color=C_HOMEO, linewidth=2.0, label='$r=16$ (N=3)', zorder=3)

# r=128 single
ax1.plot(qwen_r128_g, qwen_r128_r, color=C_BOUNDED, linewidth=2.0, linestyle='--',
         label='$r=128$ (N=1)', zorder=3)

# r=256 band
g256, mean256, min256, max256 = compute_band(qwen_r256)
ax1.fill_between(g256, min256, max256, color=C_DEG, alpha=0.15, zorder=2)
ax1.plot(g256, mean256, color=C_DEG, linewidth=2.0, label='$r=256$ (N=3)', zorder=3)

ax1.set_xlim(0, 10)
ax1.set_ylim(60, 102)
ax1.set_xticks(range(0, 11, 2))
ax1.set_ylabel('$K_0$ Retention (%)', fontsize=9)
ax1.set_title('(a) Qwen 2.5 1.5B', fontsize=9, fontweight='bold', loc='left')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
           frameon=False, fontsize=7)
ax1.grid(True, alpha=0.12, linewidth=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Regime labels on right
ax1.text(10.3, 96, 'Homeo.', fontsize=6.5, color='#065F46', va='center')
ax1.text(10.3, 85, 'Bounded', fontsize=6.5, color='#78350F', va='center')
ax1.text(10.3, 70, 'Degrad.', fontsize=6.5, color='#991B1B', va='center')

# --- Panel (b): Gemma 3 ---
add_regime_bands(ax2, ymin=55)

# r=4 band
if gemma_r4:
    g4, mean4, min4, max4 = compute_band(gemma_r4)
    ax2.fill_between(g4, min4, max4, color=C_HOMEO, alpha=0.15, zorder=2)
    ax2.plot(g4, mean4, color=C_HOMEO, linewidth=2.0,
             label=f'$r=4$ (N={len(gemma_r4)})', zorder=3)

# r=16 band
if gemma_r16:
    g16g, mean16g, min16g, max16g = compute_band(gemma_r16)
    ax2.fill_between(g16g, min16g, max16g, color=C_DEG, alpha=0.15, zorder=2)
    ax2.plot(g16g, mean16g, color=C_DEG, linewidth=2.0,
             label=f'$r=16$ (N={len(gemma_r16)})', zorder=3)

ax2.set_xlim(0, 5)
ax2.set_ylim(55, 102)
ax2.set_xticks(range(0, 6))
ax2.set_xlabel('Generation', fontsize=9)
ax2.set_ylabel('$K_0$ Retention (%)', fontsize=9)
ax2.set_title('(b) Gemma 3 1B', fontsize=9, fontweight='bold', loc='left')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
           frameon=False, fontsize=7)
ax2.grid(True, alpha=0.12, linewidth=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Regime labels on right
ax2.text(5.15, 96, 'Homeo.', fontsize=6.5, color='#065F46', va='center')
ax2.text(5.15, 85, 'Bounded', fontsize=6.5, color='#78350F', va='center')
ax2.text(5.15, 67, 'Degrad.', fontsize=6.5, color='#991B1B', va='center')

plt.subplots_adjust(right=0.84, hspace=0.50)
fig.savefig(FIG_DIR / "fig_combined_trajectories.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'fig_combined_trajectories.png'}")
plt.close()
