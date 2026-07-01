"""Regenerate ALL data figures with Plotly.

Figures:
  1. Longitudinal retention trajectories (r=16, r=128, r=256)
  2. Dose-response curve (retention + eff rank vs nominal rank)
  3. Cross-backbone comparison (retention vs eff rank)
  4. FFT vs QLoRA Gen10 trajectories (3 seeds)
  5. Distributional signatures (length, persistence, efficiency)
  6. Intervention comparison (bar chart)
  + Heatmap (already done, regenerated here for consistency)

Style: Inter font, clean white bg, consistent colors, 4x export.

Run: uv run python scripts/gen_all_plotly.py
"""
import json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

OUT = Path(__file__).parent.parent / "outputs"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

# Consistent style
FONT = "Inter, Arial, sans-serif"
COLORS = {
    "homeostatic": "#1a9641",
    "bounded": "#fdae61",
    "degradative": "#d7191c",
    "qwen": "#2166ac",
    "gemma": "#b2182b",
    "gemma4": "#762a83",
    "fft": "#e66101",
    "qlora": "#5e3c99",
    "c1": "#888888",
    "c2": "#4393c3",
    "c3": "#2166ac",
    "c5": "#053061",
}
LAYOUT_DEFAULTS = dict(
    font=dict(family=FONT, size=12),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=60, r=20, t=30, b=50),
)


def style_axes(fig, xgrid=False, ygrid=True):
    fig.update_xaxes(showgrid=xgrid, gridcolor="#eee", zeroline=False,
                     linecolor="#ccc", linewidth=1, mirror=True)
    fig.update_yaxes(showgrid=ygrid, gridcolor="#eee", zeroline=False,
                     linecolor="#ccc", linewidth=1, mirror=True)
    return fig


def save(fig, name, w=480, h=340):
    pio.write_image(fig, str(FIG_DIR / f"{name}.png"), width=w, height=h, scale=4)
    print(f"  OK: {name}.png")


# ============================================================
# FIGURE 1: Longitudinal Retention Trajectories
# ============================================================
print("Fig 1: Trajectories...")

def load_retention(folder, k0_key="k0_size"):
    data = json.loads((OUT / folder / "results.json").read_text())
    k0 = data[0][k0_key]
    gens = [r["generation"] for r in data]
    ret = [r["retention"] / k0 * 100 for r in data]
    return gens, ret

# r=16 (3 seeds)
seeds_16 = []
for s in [15, 137, 256]:
    try:
        g, r = load_retention(f"g1_rank256_seed{s}".replace("256", "16").replace("g1_rank16", "fft_drift_gen10/../g1_rank16") if False else f"fft_drift_gen10")
    except:
        pass

# Load from fft_drift_gen10 (has QLoRA r=16 data)
def load_fft_gen10(filename, k0=78):
    data = json.loads((OUT / "fft_drift_gen10" / filename).read_text())
    gens = [0] + [r["gen"] for r in data]
    ret = [100.0] + [r["retention"] / k0 * 100 for r in data]
    return gens, ret

fig1 = go.Figure()

# r=16 seeds
for s, dash in zip([15, 137, 256], [None, "dot", "dash"]):
    gens, ret = load_fft_gen10(f"qlora_r16_seed{s}.json")
    fig1.add_trace(go.Scatter(x=gens, y=ret, mode="lines+markers",
        name=f"r=16 (s{s})", line=dict(color=COLORS["homeostatic"], width=2, dash=dash),
        marker=dict(size=5), legendgroup="r16", showlegend=(s==15)))

# r=128
g128, r128 = load_retention("g1_rank128_seed15")
fig1.add_trace(go.Scatter(x=g128, y=r128, mode="lines+markers",
    name="r=128", line=dict(color=COLORS["bounded"], width=2.5),
    marker=dict(size=5)))

# r=256 seeds
for s, dash in zip([15, 137, 256], [None, "dot", "dash"]):
    g256, r256 = load_retention(f"g1_rank256_seed{s}")
    fig1.add_trace(go.Scatter(x=g256, y=r256, mode="lines+markers",
        name=f"r=256 (s{s})", line=dict(color=COLORS["degradative"], width=2, dash=dash),
        marker=dict(size=5), legendgroup="r256", showlegend=(s==15)))

fig1.update_layout(**LAYOUT_DEFAULTS, width=520, height=320,
    xaxis_title="Generation", yaxis_title="K0 Retention (%)",
    yaxis_range=[65, 102],
    legend=dict(x=0.65, y=0.95, font=dict(size=10)))
style_axes(fig1)
save(fig1, "fig1_trajectories", w=520, h=320)


# ============================================================
# FIGURE 2: Dose-Response Curve
# ============================================================
print("Fig 2: Dose-response...")

ranks = [4, 16, 32, 64, 128, 256]
retention_gen5 = [96.2, 94.9, 94.9, 89.9, 87.3, 83.1]  # Gen5 values
retention_gen10 = [None, 94.9, None, 91.1, 88.6, 78.0]  # Gen10 where available
eff_ranks = [3.34, 11.08, 17.85, 29.52, 50.16, 87.57]

fig2 = make_subplots(rows=1, cols=2, subplot_titles=("K0 Retention vs Rank", "Effective Rank vs Rank"))

fig2.add_trace(go.Scatter(x=ranks, y=retention_gen5, mode="lines+markers",
    marker=dict(size=8, color=COLORS["qwen"]), line=dict(width=2.5, color=COLORS["qwen"]),
    name="Gen5"), row=1, col=1)

fig2.add_trace(go.Scatter(x=ranks, y=eff_ranks, mode="lines+markers",
    marker=dict(size=8, color=COLORS["degradative"]), line=dict(width=2.5, color=COLORS["degradative"]),
    name="Eff. Rank"), row=1, col=2)

fig2.update_xaxes(title_text="Nominal Rank", type="log", row=1, col=1)
fig2.update_xaxes(title_text="Nominal Rank", type="log", row=1, col=2)
fig2.update_yaxes(title_text="Retention (%)", row=1, col=1)
fig2.update_yaxes(title_text="Effective Rank", row=1, col=2)
fig2.update_layout(**LAYOUT_DEFAULTS, width=580, height=300, showlegend=False)
style_axes(fig2)
save(fig2, "fig2_dose_response", w=580, h=300)


# ============================================================
# FIGURE 3: Cross-Backbone Comparison
# ============================================================
print("Fig 3: Cross-backbone...")

# (eff_rank, retention%, label)
qwen = [(3.34, 96.2), (11.08, 94.9), (17.85, 94.9), (29.52, 91.1), (50.16, 88.6), (87.57, 78.0)]
gemma = [(1.75, 97.9), (3.07, 91.3), (6.37, 78.3), (7.35, 73.9), (8.37, 69.6), (9.17, 69.1), (71.47, 70.2)]
gemma4 = [(2.13, 96.1), (5.64, 97.4)]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=[d[0] for d in qwen], y=[d[1] for d in qwen],
    mode="markers+lines", name="Qwen 2.5 1.5B",
    marker=dict(size=9, symbol="circle", color=COLORS["qwen"]),
    line=dict(width=2, color=COLORS["qwen"])))
fig3.add_trace(go.Scatter(x=[d[0] for d in gemma], y=[d[1] for d in gemma],
    mode="markers+lines", name="Gemma 3 1B",
    marker=dict(size=9, symbol="square", color=COLORS["gemma"]),
    line=dict(width=2, color=COLORS["gemma"])))
fig3.add_trace(go.Scatter(x=[d[0] for d in gemma4], y=[d[1] for d in gemma4],
    mode="markers", name="Gemma 4 E2B",
    marker=dict(size=9, symbol="triangle-up", color=COLORS["gemma4"])))

fig3.update_layout(**LAYOUT_DEFAULTS, width=500, height=340,
    xaxis_title="Mean Effective Rank", yaxis_title="K0 Retention (%)",
    xaxis_type="log", yaxis_range=[60, 102],
    legend=dict(x=0.55, y=0.95, font=dict(size=10)))
style_axes(fig3)
save(fig3, "fig3_cross_backbone", w=500, h=340)


# ============================================================
# FIGURE 4: FFT vs QLoRA Gen10
# ============================================================
print("Fig 4: FFT vs QLoRA...")

fig4 = go.Figure()
for s, dash in zip([15, 137, 256], [None, "dot", "dash"]):
    gens, ret = load_fft_gen10(f"qlora_r16_seed{s}.json")
    fig4.add_trace(go.Scatter(x=gens, y=ret, mode="lines+markers",
        name=f"QLoRA (s{s})", line=dict(color=COLORS["qlora"], width=2, dash=dash),
        marker=dict(size=4), legendgroup="qlora", showlegend=(s==15)))

for s, dash in zip([15, 137, 256], [None, "dot", "dash"]):
    gens, ret = load_fft_gen10(f"fft_lr1e-06_seed{s}.json")
    fig4.add_trace(go.Scatter(x=gens, y=ret, mode="lines+markers",
        name=f"FFT (s{s})", line=dict(color=COLORS["fft"], width=2, dash=dash),
        marker=dict(size=4), legendgroup="fft", showlegend=(s==15)))

fig4.update_layout(**LAYOUT_DEFAULTS, width=500, height=300,
    xaxis_title="Generation", yaxis_title="K0 Retention (%)",
    yaxis_range=[85, 102],
    legend=dict(x=0.65, y=0.35, font=dict(size=10)))
style_axes(fig4)
save(fig4, "fig4_fft_vs_qlora", w=500, h=300)


# ============================================================
# FIGURE 5: Distributional Signatures
# ============================================================
print("Fig 5: Distributional...")

# Data from analyze_complete_metrics (hardcoded from CSV analysis)
gens_dist = list(range(0, 11))
# r=16 (homeostatic)
len_r16 = [2.5, 2.6, 2.6, 2.6, 2.7, 2.7, 2.6, 2.6, 2.7, 2.7, 2.7]
eff_r16 = [0.41, 0.38, 0.37, 0.37, 0.37, 0.36, 0.37, 0.37, 0.36, 0.36, 0.35]
pers_r16 = [100, 35.7, 35.7, 35.7, 35.7, 35.7, 35.7, 36.1, 35.7, 36.1, 36.1]
# r=128 (bounded)
len_r128 = [2.5, 3.3, 4.1, 4.8, 5.4, 5.9, 6.2, 6.5, 6.7, 6.9, 7.0]
eff_r128 = [0.41, 0.27, 0.22, 0.18, 0.16, 0.15, 0.14, 0.14, 0.13, 0.13, 0.13]
pers_r128 = [100, 28.6, 18.4, 13.3, 10.2, 8.2, 7.1, 6.1, 5.2, 5.2, 5.2]
# r=256 (degradative)
len_r256 = [2.5, 2.8, 3.0, 3.1, 3.2, 3.3, 3.4, 3.4, 3.5, 3.5, 3.6]
eff_r256 = [0.41, 0.35, 0.31, 0.29, 0.27, 0.26, 0.24, 0.23, 0.22, 0.22, 0.22]
pers_r256 = [100, 32.6, 20.4, 14.3, 10.2, 8.2, 6.1, 5.1, 4.1, 4.1, 3.8]

fig5 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=("Mean Response Length (words)", "Content Efficiency", "Baseline Persistence (%)"))

for data, name, color in [(len_r16, "r=16", COLORS["homeostatic"]),
                           (len_r128, "r=128", COLORS["bounded"]),
                           (len_r256, "r=256", COLORS["degradative"])]:
    fig5.add_trace(go.Scatter(x=gens_dist, y=data, mode="lines+markers", name=name,
        line=dict(color=color, width=2.5), marker=dict(size=4),
        legendgroup=name, showlegend=True), row=1, col=1)

for data, name, color in [(eff_r16, "r=16", COLORS["homeostatic"]),
                           (eff_r128, "r=128", COLORS["bounded"]),
                           (eff_r256, "r=256", COLORS["degradative"])]:
    fig5.add_trace(go.Scatter(x=gens_dist, y=data, mode="lines+markers", name=name,
        line=dict(color=color, width=2.5), marker=dict(size=4),
        legendgroup=name, showlegend=False), row=2, col=1)

for data, name, color in [(pers_r16, "r=16", COLORS["homeostatic"]),
                           (pers_r128, "r=128", COLORS["bounded"]),
                           (pers_r256, "r=256", COLORS["degradative"])]:
    fig5.add_trace(go.Scatter(x=gens_dist, y=data, mode="lines+markers", name=name,
        line=dict(color=color, width=2.5), marker=dict(size=4),
        legendgroup=name, showlegend=False), row=3, col=1)

fig5.update_xaxes(title_text="Generation", row=3, col=1)
fig5.update_layout(**LAYOUT_DEFAULTS, width=480, height=520, showlegend=True,
    legend=dict(x=0.7, y=0.98, font=dict(size=10)))
style_axes(fig5)
save(fig5, "fig5_distributional", w=480, h=520)


# ============================================================
# FIGURE 6: Intervention Comparison
# ============================================================
print("Fig 6: Interventions...")

conditions = ["C1\n(baseline)", "C2\n(short)", "C3\n(filter)", "C5\n(random)"]
retention_vals = [83.3, 91.0, 92.3, 92.7]
colors_bar = [COLORS["c1"], COLORS["c2"], COLORS["c3"], COLORS["c5"]]

fig6 = go.Figure()
fig6.add_trace(go.Bar(x=conditions, y=retention_vals,
    marker_color=colors_bar, text=[f"{v:.1f}%" for v in retention_vals],
    textposition="outside", textfont=dict(size=11)))

fig6.add_hline(y=83.3, line_dash="dash", line_color="#888", line_width=1,
    annotation_text="Baseline (C1)", annotation_position="top right",
    annotation_font_size=10)

fig6.update_layout(**LAYOUT_DEFAULTS, width=420, height=320,
    xaxis_title="Intervention Condition",
    yaxis_title="K0 Retention (%, Gen5)",
    yaxis_range=[75, 100], showlegend=False)
style_axes(fig6)
save(fig6, "fig6_interventions", w=420, h=320)


# ============================================================
# HEATMAP (regenerate for consistency)
# ============================================================
print("Fig Heatmap: Rank×LR...")

ranks_h = [16, 64, 256]
lrs_h = ["5×10⁻⁶", "10⁻⁵", "2×10⁻⁵"]
ret_h = np.array([[76, 76, 72], [76, 72, 69], [70, 66, 64]])
ret_pct = ret_h / 78 * 100
text_h = np.array([["97.4%", "97.4%", "92.3%"],
                   ["97.4%", "92.3%", "88.5%"],
                   ["89.7%", "84.6%", "82.1%"]])

fig_h = go.Figure(data=go.Heatmap(
    z=ret_pct, x=lrs_h, y=[f"r = {r}" for r in ranks_h],
    text=text_h, texttemplate="%{text}",
    textfont={"size": 16, "family": FONT, "color": "#1a1a1a"},
    colorscale=[[0, "#1a9641"], [0.3, "#a6d96a"], [0.5, "#ffffbf"], [0.7, "#fdae61"], [1, "#d7191c"]],
    zmin=80, zmax=100,
    colorbar=dict(title=dict(text="Retention (%)", font=dict(size=11, family=FONT)),
                  tickfont=dict(size=10, family=FONT), thickness=12, len=0.85),
))
fig_h.update_layout(**LAYOUT_DEFAULTS, width=520, height=340,
    xaxis_title="Learning Rate", yaxis=dict(title="Adapter Rank", autorange="reversed"))
save(fig_h, "fig_rank_lr_heatmap", w=520, h=340)


print("\nAll figures regenerated with Plotly!")
