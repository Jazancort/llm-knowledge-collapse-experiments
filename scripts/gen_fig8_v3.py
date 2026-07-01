"""ETP Framework Figure — Editorial style.

Inspired by: narrative infographic with flowing lines showing regime divergence.
Left side: text legend. Right side: flowing curves that separate after threshold.
Minimal, elegant, typography-driven.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"

# Colors
C_HOMEO = "#1b3a5c"    # Dark navy for homeostatic (stable, solid)
C_BOUND = "#b87333"    # Copper/bronze for bounded
C_DEGRAD = "#c0392b"   # Muted red for degradative
C_BG = "#faf8f5"       # Warm off-white
C_GRID = "#e8e4df"     # Subtle grid
C_TEXT = "#2c2c2c"     # Near-black
C_MUTED = "#8c8578"    # Muted gray-brown
C_ACCENT = "#9b4d3a"   # Section header accent

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# === Subtle vertical grid lines ===
for x in np.linspace(35, 95, 8):
    ax.axvline(x, color=C_GRID, linewidth=0.5, alpha=0.6)

# === THRESHOLD vertical dashed line ===
thresh_x = 58
ax.axvline(thresh_x, color=C_ACCENT, linewidth=1.2, linestyle="--", alpha=0.7)
ax.text(thresh_x + 0.8, 93, "THRESHOLD", fontsize=7, fontweight="bold",
        color=C_ACCENT, fontfamily="sans-serif", ha="left", va="top")
ax.text(thresh_x + 0.8, 88, "(backbone-dependent)", fontsize=6.5,
        color=C_MUTED, fontfamily="sans-serif", ha="left", va="top", fontstyle="italic")

# === THREE FLOWING CURVES ===
x_pts = np.linspace(35, 97, 200)

# Homeostatic: starts flat, stays flat (high retention)
homeo_base = 55
homeo_y = homeo_base + 2 * np.sin(x_pts * 0.08) + np.where(x_pts > thresh_x,
    (x_pts - thresh_x) * 0.35, 0)

# Bounded: starts together, diverges slightly down, then levels
bound_y = homeo_base - 1 + np.where(x_pts > thresh_x,
    -((x_pts - thresh_x) * 0.12) + 3 * np.sin((x_pts - thresh_x) * 0.15),
    1.5 * np.sin(x_pts * 0.1))

# Degradative: starts together, drops significantly after threshold
degrad_y = homeo_base - 2 + np.where(x_pts > thresh_x,
    -((x_pts - thresh_x) * 0.55) + 2 * np.sin((x_pts - thresh_x) * 0.12),
    0.5 * np.sin(x_pts * 0.12))

# Plot curves
ax.plot(x_pts, homeo_y, color=C_HOMEO, linewidth=2.8, solid_capstyle="round", zorder=3)
ax.plot(x_pts, bound_y, color=C_BOUND, linewidth=2.2, solid_capstyle="round", zorder=2)
ax.plot(x_pts, degrad_y, color=C_DEGRAD, linewidth=2.2, solid_capstyle="round", zorder=2)

# Dots at curve endpoints (right side)
ax.plot(x_pts[-1], homeo_y[-1], "o", color=C_HOMEO, markersize=5, zorder=4)
ax.plot(x_pts[-1], bound_y[-1], "o", color=C_BOUND, markersize=5, zorder=4)
ax.plot(x_pts[-1], degrad_y[-1], "o", color=C_DEGRAD, markersize=5, zorder=4)

# Starting point (all together)
ax.plot(x_pts[0], homeo_base, "o", color=C_HOMEO, markersize=7, zorder=5)

# === LEFT SIDE: Text legend ===
# Section header
ax.text(2, 95, "PRESSURE-GATED REGIMES", fontsize=7, fontweight="bold",
        color=C_ACCENT, fontfamily="sans-serif")

# Main title
ax.text(2, 85, "Regime Divergence", fontsize=16, fontweight="bold",
        color=C_TEXT, fontfamily="sans-serif")
ax.text(2, 79, "How effective training pressure\nseparates recursive outcomes",
        fontsize=8.5, color=C_MUTED, fontfamily="sans-serif", va="top", linespacing=1.4)

# Legend items
legend_items = [
    (C_HOMEO, "Homeostatic", "stable retention, faithful output"),
    (C_BOUND, "Bounded", "retention holds, distribution drifts"),
    (C_DEGRAD, "Degradative", "progressive factual loss"),
]

for i, (color, title, desc) in enumerate(legend_items):
    y_pos = 60 - i * 10
    ax.plot(3, y_pos, "o", color=color, markersize=7)
    ax.text(6, y_pos, title, fontsize=9.5, fontweight="bold", color=color,
            fontfamily="sans-serif", va="center")
    ax.text(6, y_pos - 3.5, desc, fontsize=7.5, color=C_MUTED,
            fontfamily="sans-serif", va="center")

# === BOTTOM LEFT: Observed in ===
ax.text(2, 22, "OBSERVED IN", fontsize=6.5, fontweight="bold",
        color=C_MUTED, fontfamily="sans-serif")
ax.text(2, 17, r"$K_0$ retention $\cdot$ content efficiency", fontsize=7.5,
        color=C_MUTED, fontfamily="sans-serif")
ax.text(2, 13, r"response persistence $\cdot$ length drift", fontsize=7.5,
        color=C_MUTED, fontfamily="sans-serif")

# === BOTTOM: X-axis label ===
ax.text(75, 5, r"increasing training pressure $\rightarrow$", fontsize=8,
        color=C_MUTED, fontfamily="sans-serif", fontstyle="italic", ha="center")

# === Pressure components (subtle, below x-label) ===
ax.text(75, 1.5, "rank  |  learning rate  |  synthetic exposure", fontsize=6.5,
        color=C_GRID, fontfamily="sans-serif", ha="center")

plt.tight_layout(pad=0.3)
plt.savefig(FIG_DIR / "fig_etp_framework_v3.png", dpi=300, bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
plt.savefig(FIG_DIR / "fig_etp_framework_v3.svg", bbox_inches="tight",
            facecolor=C_BG, edgecolor="none")
print("OK: fig_etp_framework_v3.png + .svg")
