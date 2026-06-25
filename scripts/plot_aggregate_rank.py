"""Plot Retention vs Mean Effective Rank — corrected figures.

Fig 5a: Gen5 (all configs)
Fig 5b: Gen10 (only configs with Gen10 data)

Uses mean effective rank as x-axis (as measured by compute_lora_spectrum on real adapters).
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif",
})

# Data: (label, mean_eff_rank, retention_gen5, retention_gen10, marker, color)
# mean_eff_rank from compute_lora_spectrum() on real adapters (Gen1 values, stable across gens)

data = [
    # Attention-only
    ("Attn r=4",   3.34,  96.2, None,  "o", "#2196F3"),
    ("Attn r=16",  11.08, 94.9, 94.9,  "o", "#2196F3"),
    ("Attn r=32",  17.8,  94.9, None,  "o", "#2196F3"),
    ("Attn r=64",  30.1,  89.9, 91.1,  "o", "#2196F3"),
    ("Attn r=128", 50.5,  87.3, 88.6,  "o", "#2196F3"),
    ("Attn r=256", 87.8,  83.1, 78.0,  "o", "#2196F3"),
    # Full Linear
    ("Full r=4",   3.43,  94.9, None,  "s", "#E53935"),
    ("Full r=16",  11.5,  91.1, 87.3,  "s", "#E53935"),
]


def plot_fig(gen_key, title_suffix, filename):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    idx = 2 if gen_key == "gen5" else 3

    for label, eff, ret5, ret10, marker, color in data:
        val = ret5 if gen_key == "gen5" else ret10
        if val is None:
            continue
        ax.scatter(eff, val, marker=marker, s=130, c=color,
                   edgecolors='black', linewidths=0.5, zorder=5)
        offset_y = 1.2 if val > 88 else -2.0
        ax.annotate(label, (eff, val), textcoords="offset points",
                    xytext=(5, offset_y), fontsize=9, color=color)

    # Connect attention points
    attn = [(d[1], d[idx]) for d in data if d[4] == "o" and d[idx] is not None]
    attn.sort()
    ax.plot([a[0] for a in attn], [a[1] for a in attn], '--', color="#2196F3", alpha=0.4, linewidth=1.5)

    # Connect full linear points
    full = [(d[1], d[idx]) for d in data if d[4] == "s" and d[idx] is not None]
    full.sort()
    if len(full) > 1:
        ax.plot([f[0] for f in full], [f[1] for f in full], '--', color="#E53935", alpha=0.4, linewidth=1.5)

    # Legend
    ax.scatter([], [], marker='o', s=80, c='#2196F3', edgecolors='black', linewidths=0.5, label='Attention-only (q, v)')
    ax.scatter([], [], marker='s', s=80, c='#E53935', edgecolors='black', linewidths=0.5, label='Full Linear (all 7 modules)')
    ax.legend(loc='lower left', framealpha=0.9)

    ax.set_xlabel('Mean Effective Rank per Adapter (measured on real checkpoint)')
    ax.set_ylabel(f'K₀ Retention at {title_suffix} (%)')
    ax.set_title(f'Retention Tracks Mean Effective Rank Across Targeting Schemes ({title_suffix})')
    ax.set_ylim(72, 100)
    ax.set_xlim(0, 95)
    ax.grid(True, alpha=0.3)

    # Degradative zone shading
    if gen_key == "gen10":
        ax.axhspan(72, 82, color='#FFCDD2', alpha=0.2, zorder=0)
        ax.text(75, 74, 'Degradative regime', fontsize=9, color='#B71C1C', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{filename}.png")
    plt.savefig(OUTPUT_DIR / f"{filename}.pdf")
    plt.close()
    print(f"  {filename} - DONE")


print("Generating corrected figures...")
plot_fig("gen5", "Gen 5", "fig5a_retention_vs_effrank_gen5")
plot_fig("gen10", "Gen 10", "fig5b_retention_vs_effrank_gen10")
print(f"\nSaved to: {OUTPUT_DIR}")
