"""Figure: Combined trajectories — 3 panels (Qwen, Gemma 3 Gen10, E2B Gen10).
All backbones now shown over 10 generations for consistency.
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
FIG_DIR = Path(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v3\figs")
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Colors (Wong palette)
C_HOMEO = '#0072B2'
C_BOUNDED = '#E69F00'
C_DEG = '#D55E00'


def load_v3(folder):
    """Load v3 experiment results. Prepend Gen0=100%."""
    path = OUT / folder / "results.json"
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    k0 = data["k0_size"]
    gens = [0] + [g["gen"] for g in data["generations"]]
    ret = [100.0] + [g["retention"] / k0 * 100 for g in data["generations"]]
    return gens, ret


def load_qwen_qlora(filename, k0=78):
    """Load Qwen QLoRA data from fft_drift_gen10 folder."""
    path = OUT / "fft_drift_gen10" / filename
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    gens = [0] + [r["gen"] for r in data]
    ret = [100.0] + [r["retention"] / k0 * 100 for r in data]
    return gens, ret


def load_qwen_rank(folder, k0=78):
    """Load Qwen rank experiments."""
    path = OUT / folder / "results.json"
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        gens = [r["generation"] for r in data]
        ret = [r["retention"] / k0 * 100 for r in data]
    else:
        gens = [r["generation"] for r in data.get("generations", data)]
        ret = [r["retention"] / k0 * 100 for r in data.get("generations", data)]
    return gens, ret


def compute_band(trajectories):
    """Compute mean, min, max across seeds."""
    valid = [(g, r) for g, r in trajectories if len(g) > 0]
    if not valid:
        return [], [], [], []
    min_len = min(len(r) for _, r in valid)
    arr = np.array([r[:min_len] for _, r in valid])
    gens = valid[0][0][:min_len]
    return gens, arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)


def add_regime_bands(ax, ymin=45):
    """Add colored regime background bands."""
    ax.axhspan(90, 102, color='#D1FAE5', alpha=0.35, zorder=0, clip_on=True)
    ax.axhspan(80, 90, color='#FDE68A', alpha=0.25, zorder=0, clip_on=True)
    ax.axhspan(ymin, 80, color='#FECACA', alpha=0.25, zorder=0, clip_on=True)
    ax.axhline(90, color='#065F46', linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)
    ax.axhline(80, color='#991B1B', linewidth=0.4, linestyle='--', alpha=0.4, zorder=1)


# ============================================================
# LOAD DATA
# ============================================================

# Qwen (existing data)
qwen_r16 = [load_qwen_qlora(f"qlora_r16_seed{s}.json") for s in [15, 137, 256]]
qwen_r128_g, qwen_r128_r = load_qwen_rank("g1_rank128_seed15")
qwen_r256 = [load_qwen_rank(f"g1_rank256_seed{s}") for s in [15, 137, 256]]

# Gemma 3 Gen10 (v3 data)
gemma3_r4 = [load_v3(f"v3_gemma3_rank4_seed{s}") for s in [15, 137, 256, 42, 77]]
gemma3_r16 = [load_v3(f"v3_gemma3_rank16_seed{s}") for s in [15, 137, 256, 42, 77]]

# E2B Gen10 (v3 data)
e2b_r4 = load_v3("v3_gemma4_rank4_seed15")
e2b_r16 = load_v3("v3_gemma4_rank16_seed15")
e2b_r64 = load_v3("v3_gemma4_rank64_seed15")

# ============================================================
# PLOT — 3 panels horizontal (side by side, figure*)
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)

# --- Panel (a): Qwen ---
add_regime_bands(ax1)

g16, mean16, min16, max16 = compute_band(qwen_r16)
ax1.fill_between(g16, min16, max16, color=C_HOMEO, alpha=0.15, zorder=2)
ax1.plot(g16, mean16, color=C_HOMEO, linewidth=2.0, label='$r=16$', zorder=3)

ax1.plot(qwen_r128_g, qwen_r128_r, color=C_BOUNDED, linewidth=2.0, linestyle='--',
         label='$r=128$', zorder=3)

g256, mean256, min256, max256 = compute_band(qwen_r256)
ax1.fill_between(g256, min256, max256, color=C_DEG, alpha=0.15, zorder=2)
ax1.plot(g256, mean256, color=C_DEG, linewidth=2.0, label='$r=256$', zorder=3)

ax1.set_xlim(0, 10)
ax1.set_ylim(45, 102)
ax1.set_xticks(range(0, 11, 2))
ax1.set_xlabel('Generation', fontsize=8)
ax1.set_ylabel('$K_0$ Retention (%)', fontsize=8)
ax1.set_title('(a) Qwen 2.5 1.5B', fontsize=9, fontweight='bold', loc='left')
ax1.grid(True, alpha=0.12, linewidth=0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Panel (b): Gemma 3 Gen10 ---
add_regime_bands(ax2)

g4, mean4, min4, max4 = compute_band(gemma3_r4)
ax2.fill_between(g4, min4, max4, color=C_HOMEO, alpha=0.15, zorder=2)
ax2.plot(g4, mean4, color=C_HOMEO, linewidth=2.0, label='$r=4$', zorder=3)

g16g, mean16g, min16g, max16g = compute_band(gemma3_r16)
ax2.fill_between(g16g, min16g, max16g, color=C_DEG, alpha=0.15, zorder=2)
ax2.plot(g16g, mean16g, color=C_DEG, linewidth=2.0, label='$r=16$', zorder=3)

ax2.set_xlim(0, 10)
ax2.set_ylim(45, 102)
ax2.set_xticks(range(0, 11, 2))
ax2.set_xlabel('Generation', fontsize=8)
ax2.set_title('(b) Gemma 3 1B', fontsize=9, fontweight='bold', loc='left')
ax2.grid(True, alpha=0.12, linewidth=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Panel (c): E2B Gen10 ---
add_regime_bands(ax3)

if e2b_r4[0]:
    ax3.plot(e2b_r4[0], e2b_r4[1], color=C_HOMEO, linewidth=2.0, label='$r=4$', zorder=3)
if e2b_r16[0]:
    ax3.plot(e2b_r16[0], e2b_r16[1], color=C_BOUNDED, linewidth=2.0, label='$r=16$', zorder=3)
if e2b_r64[0]:
    ax3.plot(e2b_r64[0], e2b_r64[1], color=C_DEG, linewidth=2.0, label='$r=64$', zorder=3)

ax3.set_xlim(0, 10)
ax3.set_ylim(45, 102)
ax3.set_xticks(range(0, 11, 2))
ax3.set_xlabel('Generation', fontsize=8)
ax3.set_title('(c) Gemma 4 E2B', fontsize=9, fontweight='bold', loc='left')
ax3.grid(True, alpha=0.12, linewidth=0.4)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Per-panel legends (avoid color duplication across panels)
ax1.legend(loc='lower left', frameon=False, fontsize=7)
ax2.legend(loc='lower left', frameon=False, fontsize=7)
ax3.legend(loc='lower left', frameon=False, fontsize=7)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_combined_trajectories.png", bbox_inches="tight", dpi=300)
print(f"Saved {FIG_DIR / 'fig_combined_trajectories.png'}")
plt.close()
