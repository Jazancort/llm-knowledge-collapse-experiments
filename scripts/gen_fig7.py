"""
Figure 7: Experimental Methodology Overview
A visual summary of the recursive fine-tuning protocol and experimental design.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import scienceplots  # noqa: F401
from pathlib import Path

plt.style.use(["science", "no-latex"])
FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Colors
c_model = "#4C72B0"
c_data = "#55A868"
c_eval = "#C44E52"
c_axes = "#8172B2"
c_light = "#F0F0F0"
c_arrow = "#333333"

# === TOP: Models ===
ax.text(5, 5.7, "Experimental Design", fontsize=11, fontweight="bold",
        ha="center", va="center")

# === LEFT COLUMN: Recursive Protocol ===
# Base model box
ax.add_patch(FancyBboxPatch((0.3, 4.2), 2.2, 0.7, boxstyle="round,pad=0.1",
             facecolor=c_model, edgecolor="black", linewidth=0.8, alpha=0.8))
ax.text(1.4, 4.55, "Base Model\n(frozen)", fontsize=7, ha="center",
        va="center", color="white", fontweight="bold")

# Arrow down
ax.annotate("", xy=(1.4, 3.9), xytext=(1.4, 4.2),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))

# Recursive loop box
ax.add_patch(FancyBboxPatch((0.1, 2.4), 2.6, 1.5, boxstyle="round,pad=0.1",
             facecolor=c_light, edgecolor="black", linewidth=1.0))
ax.text(1.4, 3.7, "Recursive Loop (Gen 0-10)", fontsize=7,
        fontweight="bold", ha="center", va="center")

# Steps inside loop
steps = [
    "1. Generate synthetic responses",
    "2. Train new QLoRA adapter",
    "3. Replace training data",
    "4. Evaluate K0 retention",
]
for i, s in enumerate(steps):
    ax.text(0.3, 3.35 - i * 0.25, s, fontsize=5.5, ha="left", va="center",
            family="monospace")

# Loop arrow (curved, going back up)
ax.annotate("", xy=(2.5, 3.6), xytext=(2.5, 2.6),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0,
                           connectionstyle="arc3,rad=-0.5"))

# === MIDDLE COLUMN: Three Pressure Axes ===
ax.add_patch(FancyBboxPatch((3.2, 2.4), 3.2, 2.5, boxstyle="round,pad=0.1",
             facecolor="#F8F4FF", edgecolor=c_axes, linewidth=1.2))
ax.text(4.8, 4.7, "Effective Training Pressure", fontsize=8,
        fontweight="bold", ha="center", va="center", color=c_axes)

# Three axes
axes_data = [
    ("Axis 1: Update Capacity", "QLoRA rank r={4,16,...,256}", 4.2),
    ("Axis 2: Perturbation Magnitude", "FFT LR={1e-6,...,2e-5}", 3.6),
    ("Axis 3: Synthetic Exposure", "C1-C5 interventions (~5%)", 3.0),
]
for label, detail, y in axes_data:
    ax.text(3.5, y, label, fontsize=6.5, ha="left", va="center",
            fontweight="bold")
    ax.text(3.5, y - 0.22, detail, fontsize=5.5, ha="left", va="center",
            color="#555555")

# === RIGHT COLUMN: Backbones + Evaluation ===
ax.add_patch(FancyBboxPatch((6.8, 3.8), 2.9, 1.2, boxstyle="round,pad=0.1",
             facecolor="#E8F5E9", edgecolor=c_data, linewidth=0.8))
ax.text(8.25, 4.8, "Backbones", fontsize=7, fontweight="bold",
        ha="center", va="center", color=c_data)
backbones = ["Qwen 2.5 1.5B (primary, N=3)", "Gemma 3 1B (secondary, N=3)",
             "Gemma 4 E2B (robustness, N=1)"]
for i, b in enumerate(backbones):
    ax.text(7.0, 4.5 - i * 0.25, b, fontsize=5.5, ha="left", va="center")

# Evaluation box
ax.add_patch(FancyBboxPatch((6.8, 2.2), 2.9, 1.3, boxstyle="round,pad=0.1",
             facecolor="#FFEBEE", edgecolor=c_eval, linewidth=0.8))
ax.text(8.25, 3.3, "Evaluation Metrics", fontsize=7, fontweight="bold",
        ha="center", va="center", color=c_eval)
metrics = ["K0 factual retention", "Content efficiency",
           "Baseline persistence", "Effective rank (erank)",
           "SDI-3, Distinct-n, MTLD"]
for i, m in enumerate(metrics):
    ax.text(7.0, 3.0 - i * 0.2, m, fontsize=5.5, ha="left", va="center")

# === BOTTOM: Dataset ===
ax.add_patch(FancyBboxPatch((2.5, 0.3), 5.0, 0.8, boxstyle="round,pad=0.1",
             facecolor="#FFF3E0", edgecolor="#E65100", linewidth=0.8))
ax.text(5.0, 0.7, "TriviaQA: 2000 train / 200 eval / K0 subset (78-79 items)",
        fontsize=6.5, ha="center", va="center", fontweight="bold")
ax.text(5.0, 0.45, "Replace-only protocol | 2 epochs | LR=1e-5 | T=0.7 | Seeds: 15, 137, 256",
        fontsize=5.5, ha="center", va="center", color="#555555")

# === BOTTOM: Output (regime classification) ===
ax.add_patch(FancyBboxPatch((0.3, 1.2), 9.4, 0.7, boxstyle="round,pad=0.1",
             facecolor="#E3F2FD", edgecolor=c_model, linewidth=0.8))
ax.text(5.0, 1.55, "Output: Regime Classification", fontsize=7,
        fontweight="bold", ha="center", va="center", color=c_model)

# Three regimes
regime_x = [2.0, 5.0, 8.0]
regime_labels = ["Homeostatic\n(stable)", "Bounded\n(drift only)",
                 "Degradative\n(factual loss)"]
regime_colors = ["#2E7D32", "#F57C00", "#C62828"]
for x, label, color in zip(regime_x, regime_labels, regime_colors):
    ax.text(x, 1.35, label, fontsize=6, ha="center", va="center",
            color=color, fontweight="bold")

# Connecting arrows
# Protocol -> Axes
ax.annotate("", xy=(3.2, 3.5), xytext=(2.7, 3.5),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0))
# Axes -> Evaluation
ax.annotate("", xy=(6.8, 3.2), xytext=(6.4, 3.2),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0))
# Evaluation -> Regimes
ax.annotate("", xy=(5.0, 1.9), xytext=(5.0, 2.2),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0))

plt.tight_layout(pad=0.2)
fig.savefig(FIG_DIR / "fig7_methodology.png", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig7_methodology.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig7_methodology.svg", bbox_inches="tight")
print("Saved fig7_methodology.png/.pdf/.svg")
plt.close()
