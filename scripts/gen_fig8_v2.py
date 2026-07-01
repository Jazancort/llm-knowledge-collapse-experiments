"""ETP Framework Figure — Redesigned.

Style: Horizontal flow, minimal, inspired by Nature/Science infographics.
No boxes-in-boxes. Uses negative space, typography hierarchy, and color bands.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")

# Fonts
TITLE_FONT = {"fontsize": 11, "fontweight": "bold", "fontfamily": "sans-serif"}
SUB_FONT = {"fontsize": 8.5, "fontfamily": "sans-serif", "color": "#555"}
SMALL_FONT = {"fontsize": 7.5, "fontfamily": "sans-serif", "color": "#777"}

# === TOP: Three pressure inputs as minimal circles with connecting lines ===
# Funnel design: 3 inputs converge into 1 output

input_y = 88
inputs = [
    (20, "Adapter\nRank", "Update\nCapacity", "#3b7dd8"),
    (50, "Learning\nRate", "Perturbation\nMagnitude", "#e67e22"),
    (80, "Synthetic\nExposure", "Training\nExamples", "#27ae60"),
]

for x, title, subtitle, color in inputs:
    # Circle
    circle = plt.Circle((x, input_y), 7, facecolor=color, edgecolor="none", alpha=0.12)
    ax.add_patch(circle)
    circle_border = plt.Circle((x, input_y), 7, facecolor="none", edgecolor=color, linewidth=1.8, alpha=0.6)
    ax.add_patch(circle_border)
    # Icon dot
    ax.plot(x, input_y, "o", color=color, markersize=6, zorder=5)
    # Text
    ax.text(x, input_y + 10.5, title, ha="center", va="center", color=color, **TITLE_FONT)
    ax.text(x, input_y - 10, subtitle, ha="center", va="center", **SMALL_FONT)

# === Converging lines to center ===
center_x, funnel_y = 50, 68
for x, _, _, color in inputs:
    ax.annotate("", xy=(center_x, funnel_y + 3), xytext=(x, input_y - 7),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2, alpha=0.5,
                                connectionstyle="arc3,rad=0"))

# === CENTRAL: Pressure indicator (gauge-style) ===
# Horizontal gradient bar representing pressure scale
bar_y = 65
bar_h = 6
bar_w = 50
bar_x = 25

# Gradient background
for i in range(200):
    frac = i / 200
    r = 0.1 + frac * 0.75
    g = 0.65 - frac * 0.5
    b = 0.25 - frac * 0.1
    ax.axvspan(bar_x + frac * bar_w, bar_x + (frac + 0.005) * bar_w,
               ymin=(bar_y - 1) / 100, ymax=(bar_y + bar_h + 1) / 100,
               color=(min(r, 1), max(g, 0), max(b, 0)), alpha=0.85)

# Bar border
bar_rect = FancyBboxPatch((bar_x, bar_y), bar_w, bar_h,
                           boxstyle="round,pad=0,rounding_size=1.5",
                           facecolor="none", edgecolor="#444", linewidth=1.2)
ax.add_patch(bar_rect)

# Label
ax.text(50, bar_y + bar_h + 3.5, "Effective Training Pressure", ha="center", va="center",
        fontsize=12, fontweight="bold", fontfamily="sans-serif", color="#222")
ax.text(bar_x - 1, bar_y + bar_h / 2, "Low", ha="right", va="center", **SUB_FONT)
ax.text(bar_x + bar_w + 1, bar_y + bar_h / 2, "High", ha="left", va="center", **SUB_FONT)

# === THRESHOLD: Vertical dashed line on the bar ===
thresh_x = 52
ax.plot([thresh_x, thresh_x], [bar_y - 2, bar_y + bar_h + 2], "--",
        color="#c0392b", linewidth=2, alpha=0.8)
ax.text(thresh_x, bar_y - 4.5, "Threshold\n(backbone-dependent)",
        ha="center", va="center", fontsize=7.5, color="#c0392b",
        fontfamily="sans-serif", fontstyle="italic")

# === BOTTOM: Three regimes as horizontal bands ===
regime_y = 32
regime_h = 16
regimes = [
    (25, 24, "Homeostatic", "#eef8f1", "#1a9641",
     ["Retention >90%", "Distribution stable", "Fully recoverable"]),
    (50, 24, "Bounded", "#fff8e6", "#c68c00",
     ["Retention stable", "Distribution drifts", "Hidden degradation"]),
    (75, 24, "Degradative", "#fff0f1", "#c0392b",
     ["Retention declines", "Distribution drifts", "Progressive loss"]),
]

for cx, w, title, bg, color, lines in regimes:
    x0 = cx - w / 2
    box = FancyBboxPatch((x0, regime_y), w, regime_h,
                          boxstyle="round,pad=0,rounding_size=1.5",
                          facecolor=bg, edgecolor=color, linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    # Top colored strip
    strip = FancyBboxPatch((x0, regime_y + regime_h - 3.5), w, 3.5,
                            boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor=color, edgecolor="none", alpha=0.15)
    ax.add_patch(strip)
    # Title
    ax.text(cx, regime_y + regime_h - 2, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=color, fontfamily="sans-serif")
    # Description lines
    for i, line in enumerate(lines):
        ax.text(cx, regime_y + regime_h - 5.5 - i * 3.2, line,
                ha="center", va="center", fontsize=7.5, color="#444", fontfamily="sans-serif")

# === Arrows from bar to regimes ===
# Left arrow (low pressure → homeostatic)
ax.annotate("", xy=(25, regime_y + regime_h), xytext=(35, bar_y),
            arrowprops=dict(arrowstyle="-|>", color="#1a9641", lw=1.3, alpha=0.6,
                            connectionstyle="arc3,rad=0.2"))
# Right arrow (high pressure → degradative)
ax.annotate("", xy=(75, regime_y + regime_h), xytext=(65, bar_y),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.3, alpha=0.6,
                            connectionstyle="arc3,rad=-0.2"))
# Middle arrow (threshold → bounded)
ax.annotate("", xy=(50, regime_y + regime_h), xytext=(50, bar_y),
            arrowprops=dict(arrowstyle="-|>", color="#c68c00", lw=1.3, alpha=0.6))

# === BOTTOM: Signatures bar ===
sig_y = 18
ax.plot([15, 85], [sig_y, sig_y], "-", color="#ddd", linewidth=0.8)
ax.text(50, sig_y - 3, "Observable signatures:", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#555", fontfamily="sans-serif")
ax.text(50, sig_y - 7,
        "K\u2080 retention  \u2022  Content efficiency  \u2022  Response persistence  \u2022  Output length drift",
        ha="center", va="center", fontsize=7.5, color="#777", fontfamily="sans-serif")

# === Section label ===
ax.text(50, 4, "Monitoring signals for recursive training governance",
        ha="center", va="center", fontsize=7, color="#999", fontfamily="sans-serif", fontstyle="italic")

plt.tight_layout(pad=0.5)
plt.savefig(FIG_DIR / "fig_etp_framework_v2.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("OK: fig_etp_framework_v2.png")
