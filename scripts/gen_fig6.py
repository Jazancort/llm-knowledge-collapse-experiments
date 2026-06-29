"""
Figure 6: Intervention Comparison
Panel A: K0 Retention at Gen5 by condition (bar chart)
Panel B: Delta Retention vs C1 baseline (bar chart)
C3 and C5 with error bars from multi-seed/multi-mask.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from pathlib import Path

plt.style.use(["science", "ieee", "no-latex"])

FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

K0 = 78

# Data from SESSION_STATE (confirmed values, Gen5)
# C1: baseline r=256 seed15 = 65/78
# C2: short-constrained seed15 = 71/78
# C3: length-filtered seeds 15/137/256 = 73, 71, 72
# C4: canonical seed15 = 69/78
# C5: random masks 15/42/99 = 72, 73, 72

conditions = ["C1\nNormal", "C2\nShort", "C3\nFiltered", "C4\nCanonical", "C5\nRandom"]
ret_mean = [65/K0*100, 71/K0*100, np.mean([73,71,72])/K0*100, 69/K0*100, np.mean([72,73,72])/K0*100]
# Error bars (range/2 for N=3 conditions, 0 for N=1)
ret_err = [0, 0, (73-71)/K0*100/2, 0, (73-72)/K0*100/2]

# Delta vs C1
c1_ret = 65/K0*100
delta_mean = [r - c1_ret for r in ret_mean]
delta_err = ret_err  # same absolute error

# Colors
colors = ["C7", "C0", "C2", "C1", "C4"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

x = np.arange(len(conditions))
width = 0.6

# Panel A: Absolute retention
bars1 = ax1.bar(x, ret_mean, width, color=colors, edgecolor="black",
                linewidth=0.5, yerr=ret_err, capsize=3, error_kw={"linewidth": 1})
ax1.axhline(y=c1_ret, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
ax1.set_ylabel("K0 Retention at Gen5 (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=7)
ax1.set_ylim(75, 100)
ax1.grid(True, alpha=0.3, axis="y")
ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontsize=9,
         va="top", ha="left")

# Panel B: Delta retention
bars2 = ax2.bar(x, delta_mean, width, color=colors, edgecolor="black",
                linewidth=0.5, yerr=delta_err, capsize=3, error_kw={"linewidth": 1})
ax2.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
ax2.set_ylabel("$\\Delta$ Retention vs C1 (pp)")
ax2.set_xticks(x)
ax2.set_xticklabels(conditions, fontsize=7)
ax2.set_ylim(-2, 14)
ax2.grid(True, alpha=0.3, axis="y")
ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, fontsize=9,
         va="top", ha="left")

# Annotate N=3 on C3 and C5
for i in [2, 4]:
    ax1.text(i, ret_mean[i] + ret_err[i] + 0.8, "N=3", ha="center",
             fontsize=6, color="black")

plt.tight_layout()

# Save
fig.savefig(FIG_DIR / "fig6_interventions.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig6_interventions.svg", bbox_inches="tight")
print("Saved fig6_interventions.pdf and .svg")
plt.close()
