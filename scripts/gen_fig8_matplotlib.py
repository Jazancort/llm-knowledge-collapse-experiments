"""Generate ETP Framework diagram (Figure 8) using Matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_PATH = Path(
    "G:/Lab/Labcity/LLM/Artigo/Paradoxo - springer/code/paper/figs/fig_etp_framework_mpl.png"
)


def draw_rounded_box(ax, x, y, w, h, facecolor="white", edgecolor="black", lw=1.2):
    """Draw a rounded rectangle and return its patch."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, start, end, color="#444444", lw=1.5):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->,head_width=6,head_length=5",
        color=color,
        linewidth=lw,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)
    return arrow


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # === LAYER 1: Three component boxes at top ===
    top_y = 10.8
    box_w, box_h = 2.6, 1.2
    components = [
        ("Adapter Rank", "Update Capacity", 1.8),
        ("Learning Rate", "Perturbation Magnitude", 5.0),
        ("Training Examples", "Synthetic Exposure", 8.2),
    ]

    for title, subtitle, cx in components:
        draw_rounded_box(
            ax, cx, top_y, box_w, box_h, facecolor="#f7f9fc", edgecolor="#5a7da8", lw=1.4
        )
        ax.text(
            cx, top_y + 0.15, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1a2a3a"
        )
        ax.text(
            cx, top_y - 0.28, subtitle, ha="center", va="center",
            fontsize=9, fontstyle="italic", color="#555555"
        )

    # === Arrows from top boxes to ETP pill ===
    etp_y = 9.0
    for _, _, cx in components:
        draw_arrow(ax, (cx, top_y - box_h / 2), (5.0, etp_y + 0.45), color="#5a7da8")

    # === LAYER 2: Central ETP pill ===
    pill_w, pill_h = 5.0, 0.9
    etp_box = FancyBboxPatch(
        (5.0 - pill_w / 2, etp_y - pill_h / 2),
        pill_w,
        pill_h,
        boxstyle="round,pad=0.03",
        facecolor="#243447",
        edgecolor="#243447",
        linewidth=2,
    )
    ax.add_patch(etp_box)
    ax.text(
        5.0, etp_y, "Effective Training Pressure",
        ha="center", va="center", fontsize=13, fontweight="bold", color="white"
    )

    # === Arrow from ETP to threshold ===
    threshold_y = 7.6
    draw_arrow(ax, (5.0, etp_y - pill_h / 2), (5.0, threshold_y + 0.2), color="#444444")

    # === LAYER 3: Threshold dashed line ===
    ax.plot(
        [1.0, 9.0], [threshold_y, threshold_y],
        linestyle="--", linewidth=2.0, color="#c0392b", zorder=3
    )
    ax.text(
        5.0, threshold_y + 0.3, "BACKBONE-DEPENDENT THRESHOLD",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="#c0392b", fontstyle="italic"
    )

    # === Arrow from threshold to regimes ===
    regime_y = 6.2
    draw_arrow(ax, (5.0, threshold_y - 0.15), (5.0, regime_y + 0.55), color="#444444")

    # === LAYER 4: Three regime boxes ===
    regime_w, regime_h = 2.6, 1.0
    regimes = [
        ("Homeostatic", "#eef8f1", "#1a9641", 1.8),
        ("Bounded", "#fff8e6", "#fdae61", 5.0),
        ("Degradative", "#fff0f1", "#d7191c", 8.2),
    ]

    for label, bg, border, cx in regimes:
        draw_rounded_box(ax, cx, regime_y, regime_w, regime_h, facecolor=bg, edgecolor=border, lw=2)
        ax.text(
            cx, regime_y, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=border
        )

    # === Arrow from regimes to observable signatures ===
    obs_y = 4.5
    for _, _, _, cx in regimes:
        draw_arrow(ax, (cx, regime_y - regime_h / 2), (5.0, obs_y + 0.55), color="#666666")

    # === LAYER 5: Observable Signatures box ===
    obs_w, obs_h = 7.5, 1.1
    draw_rounded_box(ax, 5.0, obs_y, obs_w, obs_h, facecolor="#f5f5f5", edgecolor="#333333", lw=1.5)
    ax.text(
        5.0, obs_y + 0.2, "Observable Signatures",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#1a1a1a"
    )
    ax.text(
        5.0, obs_y - 0.22,
        r"$K_0$ retention  |  Content efficiency  |  Response persistence  |  Output length drift",
        ha="center", va="center", fontsize=9.5, color="#444444"
    )

    # === Save ===
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
