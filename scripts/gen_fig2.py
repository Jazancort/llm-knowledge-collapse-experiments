"""
Figure 2: Dose-Response Curve
Panel A: K0 Retention vs Nominal Rank
Panel B: Mean Effective Rank vs Nominal Rank
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from pathlib import Path

plt.style.use(["science", "ieee", "no-latex"])

OUT = Path(__file__).parent.parent / "outputs"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

# Data (from results.json and per_matrix_rank.json)
ranks = [4, 16, 32, 64, 128, 256]
# Retention at last available gen (Gen5 for r=4/32, Gen10 for others)
retention = [96.2, 94.9, 94.9, 91.1, 88.6, 78.0]  # mean for N=3 at r=256
# Multi-seed ranges for r=16 and r=256
ret_r16_seeds = [94.9, 97.4, 97.4]  # seeds 15, 137, 256 (from fft_drift_gen10)
ret_r256_seeds = [75.9, 77.2, 81.0]  # seeds 15, 137, 256 (Gen10)
# Effective ranks (mean per-matrix)
eff_ranks = [3.34, 11.08, 17.85, 29.52, 50.16, 87.57]

# Compute error bars (range/2 for seeds with N=3)
ret_err = [0] * 6
ret_err[1] = (max(ret_r16_seeds) - min(ret_r16_seeds)) / 2  # r=16
ret_err[5] = (max(ret_r256_seeds) - min(ret_r256_seeds)) / 2  # r=256

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)

# Panel A: Retention vs Rank
ax1.errorbar(ranks, retention, yerr=ret_err, fmt="o-", color="C0",
             markersize=5, linewidth=1.5, capsize=3, capthick=1)
# Highlight N=3 points
for i in [1, 5]:
    ax1.plot(ranks[i], retention[i], "o", color="C0", markersize=7,
             markeredgecolor="black", markeredgewidth=0.5, zorder=5)
ax1.set_ylabel("K0 Retention (%)")
ax1.set_ylim(70, 100)
ax1.grid(True, alpha=0.3)
ax1.text(0.95, 0.05, "(a)", transform=ax1.transAxes, fontsize=9,
         verticalalignment="bottom", horizontalalignment="right")

# Panel B: Effective Rank vs Nominal Rank
ax2.plot(ranks, eff_ranks, "s-", color="C1", markersize=5, linewidth=1.5)
ax2.set_xlabel("Nominal LoRA Rank")
ax2.set_ylabel("Mean Effective Rank")
ax2.set_xscale("log", base=2)
ax2.set_xticks(ranks)
ax2.set_xticklabels([str(r) for r in ranks])
ax2.grid(True, alpha=0.3)
ax2.text(0.95, 0.05, "(b)", transform=ax2.transAxes, fontsize=9,
         verticalalignment="bottom", horizontalalignment="right")

# Dashed diagonal (identity: eff_rank = nominal rank)
ax2.plot([4, 256], [4, 256], "k--", alpha=0.3, linewidth=0.8, label="Identity")
ax2.legend(fontsize=6, loc="upper left")

plt.tight_layout()

# Save
fig.savefig(FIG_DIR / "fig2_dose_response.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig2_dose_response.svg", bbox_inches="tight")
print("Saved fig2_dose_response.pdf and .svg")
plt.close()
