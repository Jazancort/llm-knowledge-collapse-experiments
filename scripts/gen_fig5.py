"""
Figure 5: Distributional Signatures of Degradation
Three panels (vertical): Mean Length, Persistence, Content Efficiency
Configs: r=16 (homeostatic), r=128 (bounded), r=256 (degradative)
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from pathlib import Path

plt.style.use(["science", "ieee", "no-latex"])

FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

# Load data
data = list(csv.DictReader(
    open(Path(__file__).parent.parent / "outputs" / "diversity_analysis" / "complete_metrics.csv")
))

# Extract per config
configs_map = {
    "g1_gen10_seed15": ("$r=16$", "C0", "-"),
    "g1_rank128_seed15": ("$r=128$", "C1", "--"),
    "g1_rank256_seed15": ("$r=256$", "C3", "-."),
}

config_data = {}
for cfg, (label, color, ls) in configs_map.items():
    rows = [r for r in data if r["config"] == cfg]
    rows.sort(key=lambda r: int(r["gen"]))
    gens = [int(r["gen"]) for r in rows]
    length = [float(r["mean_length"]) for r in rows]
    persistence = [float(r["gen0_persistence"]) if r["gen0_persistence"] else None for r in rows]
    efficiency = [float(r["content_efficiency"]) for r in rows]
    config_data[cfg] = {
        "gens": gens, "length": length,
        "persistence": persistence, "efficiency": efficiency,
        "label": label, "color": color, "ls": ls,
    }

# Plot: 3 vertical panels
fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.5), sharex=True)

# Panel A: Mean Response Length
ax = axes[0]
for cfg, d in config_data.items():
    ax.plot(d["gens"], d["length"], color=d["color"], linestyle=d["ls"],
            linewidth=1.5, label=d["label"])
ax.set_ylabel("Mean Length (words)")
ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.92, "(a)", transform=ax.transAxes, fontsize=9,
        va="top", ha="right")

# Panel B: Baseline Response Persistence
ax = axes[1]
for cfg, d in config_data.items():
    # Skip Gen0 (no persistence defined)
    valid = [(g, p) for g, p in zip(d["gens"], d["persistence"]) if p is not None]
    if valid:
        gs, ps = zip(*valid)
        ax.plot(gs, [p * 100 for p in ps], color=d["color"], linestyle=d["ls"],
                linewidth=1.5, label=d["label"])
ax.set_ylabel("Gen0 Persistence (%)")
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.92, "(b)", transform=ax.transAxes, fontsize=9,
        va="top", ha="right")

# Panel C: Content Efficiency
ax = axes[2]
for cfg, d in config_data.items():
    ax.plot(d["gens"], d["efficiency"], color=d["color"], linestyle=d["ls"],
            linewidth=1.5, label=d["label"])
ax.set_ylabel("Content Efficiency")
ax.set_xlabel("Generation")
ax.set_xlim(0, 10)
ax.set_xticks(range(0, 11, 2))
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.92, "(c)", transform=ax.transAxes, fontsize=9,
        va="top", ha="right")

plt.tight_layout()

# Save
fig.savefig(FIG_DIR / "fig5_distributional.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig5_distributional.svg", bbox_inches="tight")
print("Saved fig5_distributional.pdf and .svg")
plt.close()
