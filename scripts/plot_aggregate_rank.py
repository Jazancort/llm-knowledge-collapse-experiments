"""Plot Retention vs Aggregate Effective Rank — the central figure."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 10, "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif",
})

# Aggregate effective rank data (extracted per-matrix, summed across all layers and modules)
# Format: (label, aggregate_eff_rank, retention_gen5, retention_gen10, marker, color)

data = [
    # Attention-only configs
    ("Attn r=4",   186.9,  96.2, None,  "o", "#2196F3"),
    ("Attn r=16",  620.2,  94.9, 94.9,  "o", "#2196F3"),
    ("Attn r=32",  999.3,  94.9, None,  "o", "#2196F3"),
    ("Attn r=64",  1653.2, 89.9, None,  "o", "#2196F3"),
    ("Attn r=128", 2808.9, 87.3, 88.6,  "o", "#2196F3"),
    ("Attn r=256", 4904.1, 83.1, 78.0,  "o", "#2196F3"),
    # Full Linear configs
    ("Full r=4",   671.7,  94.9, None,  "s", "#E53935"),
    ("Full r=16",  2247.7, 91.1, 87.3,  "s", "#E53935"),
]

fig, ax = plt.subplots(figsize=(9, 6))

# Plot Gen 5 points
for label, agg, ret5, ret10, marker, color in data:
    ax.scatter(agg, ret5, marker=marker, s=120, c=color, edgecolors='black', linewidths=0.5, zorder=5)
    offset_y = 1.5 if ret5 > 90 else -2.5
    offset_x = 50
    ax.annotate(label, (agg, ret5), textcoords="offset points",
                xytext=(offset_x, offset_y), fontsize=9, color=color)

# Connect attention points with a line
attn_x = [d[1] for d in data if d[4] == "o"]
attn_y = [d[2] for d in data if d[4] == "o"]
ax.plot(attn_x, attn_y, '--', color="#2196F3", alpha=0.4, linewidth=1.5)

# Connect full linear points
full_x = [d[1] for d in data if d[4] == "s"]
full_y = [d[2] for d in data if d[4] == "s"]
ax.plot(full_x, full_y, '--', color="#E53935", alpha=0.4, linewidth=1.5)

# Legend
ax.scatter([], [], marker='o', s=80, c='#2196F3', edgecolors='black', linewidths=0.5, label='Attention-only')
ax.scatter([], [], marker='s', s=80, c='#E53935', edgecolors='black', linewidths=0.5, label='Full Linear (Attn+MLP)')
ax.legend(loc='lower left', framealpha=0.9)

ax.set_xlabel('Aggregate Effective Rank (sum across all adapted matrices)')
ax.set_ylabel('K₀ Retention at Gen 5 (%)')
ax.set_title('Retention vs Aggregate Effective Rank\n(Attention-only and Full-Linear on the same curve)')
ax.set_ylim(75, 100)
ax.grid(True, alpha=0.3)

# Shade the degradative zone
ax.axhspan(75, 85, color='#FFCDD2', alpha=0.3, zorder=0)
ax.text(4500, 77, 'Degradative\nregime', fontsize=9, color='#B71C1C', alpha=0.7, ha='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig5_retention_vs_aggregate_rank.png")
plt.savefig(OUTPUT_DIR / "fig5_retention_vs_aggregate_rank.pdf")
plt.close()
print("Fig 5: Retention vs Aggregate Effective Rank - DONE")

# Print the table
print("\n" + "=" * 90)
print(f"{'Config':<15} {'Target':<12} {'Rank':<6} {'Params':<10} {'#Matrices':<10} {'Agg Eff Rank':<14} {'Ret Gen5':<10} {'Ret Gen10'}")
print("-" * 90)
table = [
    ("Attn r=4",    "attention",    4,   "545K",   56,  186.9,  96.2, None),
    ("Attn r=16",   "attention",   16,  "2.2M",   56,  620.2,  94.9, 94.9),
    ("Attn r=32",   "attention",   32,  "4.4M",   56,  999.3,  94.9, None),
    ("Full r=4",    "full_linear",  4,  "4.6M",  196,  671.7,  94.9, None),
    ("Attn r=64",   "attention",   64,  "8.7M",   56, 1653.2,  89.9, None),
    ("Full r=16",   "full_linear", 16, "18.5M",  196, 2247.7,  91.1, 87.3),
    ("Attn r=128",  "attention",  128, "17.4M",   56, 2808.9,  87.3, 88.6),
    ("Attn r=256",  "attention",  256, "34.9M",   56, 4904.1,  83.1, 78.0),
]
for row in sorted(table, key=lambda x: x[5]):
    r10 = f"{row[7]:.1f}%" if row[7] else "—"
    print(f"{row[0]:<15} {row[1]:<12} {row[2]:<6} {row[3]:<10} {row[4]:<10} {row[5]:<14.1f} {row[6]:<10.1f}% {r10}")
print("=" * 90)
