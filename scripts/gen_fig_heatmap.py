"""Generate Rank × LR heatmap figure using Plotly."""
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# Data from rank_lr_matrix experiment (Qwen 2.5 1.5B, K0=78, Gen5, seed 15)
ranks = [16, 64, 256]
lrs = ["5×10⁻⁶", "10⁻⁵", "2×10⁻⁵"]
retention = np.array([
    [76, 76, 72],   # r=16
    [76, 72, 69],   # r=64
    [70, 66, 64],   # r=256
])
retention_pct = retention / 78 * 100

# Regime labels
regime_text = np.array([
    ["97.4%", "97.4%", "92.3%"],
    ["97.4%", "92.3%", "88.5%"],
    ["89.7%", "84.6%", "82.1%"],
])

# Color scale: green (homeostatic) → yellow (bounded) → red (degradative)
colorscale = [
    [0.0, "#1a9641"],
    [0.3, "#a6d96a"],
    [0.5, "#ffffbf"],
    [0.7, "#fdae61"],
    [1.0, "#d7191c"],
]

fig = go.Figure(data=go.Heatmap(
    z=retention_pct,
    x=lrs,
    y=[f"r = {r}" for r in ranks],
    text=regime_text,
    texttemplate="%{text}",
    textfont={"size": 16, "family": "Inter, Arial", "color": "#1a1a1a"},
    colorscale=colorscale,
    zmin=80,
    zmax=100,
    colorbar=dict(
        title=dict(text="Gen5 Retention (%)", font=dict(size=12, family="Inter, Arial")),
        tickfont=dict(size=11, family="Inter, Arial"),
        thickness=14,
        len=0.85,
    ),
    hovertemplate="Rank: %{y}<br>LR: %{x}<br>Retention: %{text}<extra></extra>",
))

fig.update_layout(
    title=None,
    xaxis=dict(
        title=dict(text="Learning Rate", font=dict(size=13, family="Inter, Arial", color="#333")),
        tickfont=dict(size=12, family="Inter, Arial"),
        side="bottom",
    ),
    yaxis=dict(
        title=dict(text="Adapter Rank", font=dict(size=13, family="Inter, Arial", color="#333")),
        tickfont=dict(size=12, family="Inter, Arial"),
        autorange="reversed",
    ),
    width=520,
    height=340,
    margin=dict(l=80, r=80, t=30, b=60),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, Arial"),
)

# Export
pio.write_image(fig, "paper/figs/fig_rank_lr_heatmap.png", scale=4)
pio.write_image(fig, "paper/figs/fig_rank_lr_heatmap.svg")
print("OK: paper/figs/fig_rank_lr_heatmap.png + .svg")
