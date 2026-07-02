"""Generate Gemma 3 trajectory figure using Plotly (matches paper style)."""
import json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

OUT = Path(__file__).parent.parent / "outputs"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

FONT = "Inter, Arial, sans-serif"
LAYOUT_DEFAULTS = dict(
    font=dict(family=FONT, size=12),
    paper_bgcolor="white",
    plot_bgcolor="white",
)


def style_axes(fig):
    fig.update_xaxes(showgrid=False, gridcolor="#eee", zeroline=False,
                     linecolor="#ccc", linewidth=1, mirror=True)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False,
                     linecolor="#ccc", linewidth=1, mirror=True)
    return fig


def load_gemma3_trajectory(path):
    """Load trajectory from different JSON formats."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and 'generations' in data:
        k0 = data.get('k0_size', 47)
        gens = [g['gen'] for g in data['generations']]
        ret = [g['retention'] / k0 * 100 for g in data['generations']]
        return gens, ret
    elif isinstance(data, list) and data:
        if 'generation' in data[0]:
            k0 = data[0].get('k0_size', 47)
            gens = [g['generation'] for g in data if isinstance(g, dict) and g.get('generation', 0) > 0]
            ret = [g['retention'] / k0 * 100 for g in data if isinstance(g, dict) and g.get('generation', 0) > 0]
            return gens, ret
    return [], []


# Colors
C_HOMEO = "#1a9641"
C_DEG_LIGHT = "#fdae61"
C_DEG_MED = "#e66101"
C_DEG_DARK = "#d7191c"
C_DEG_DEEP = "#4A148C"

# Load r=4 (5 seeds)
r4_trajs = []
for seed in [15, 137, 256]:
    path = OUT / f'gemma3_rank4_seed{seed}' / 'results.json'
    if path.exists():
        r4_trajs.append(load_gemma3_trajectory(path))
for seed in [42, 77]:
    path = OUT / 'gemma3_extra_seeds' / f'gemma3_rank4_seed{seed}.json'
    if path.exists():
        r4_trajs.append(load_gemma3_trajectory(path))

# Load r=16 (5 seeds)
r16_trajs = []
for seed in [15, 137, 256]:
    path = OUT / f'gemma3_rank16_seed{seed}' / 'results.json'
    if path.exists():
        r16_trajs.append(load_gemma3_trajectory(path))
for seed in [42, 77]:
    path = OUT / 'gemma3_extra_seeds' / f'gemma3_rank16_seed{seed}.json'
    if path.exists():
        r16_trajs.append(load_gemma3_trajectory(path))

# Load intermediates (single seed)
intermediates = {}
for rank in [10, 12, 14]:
    path = OUT / 'gemma3_intermediate_ranks' / f'gemma3_rank{rank}_seed15.json'
    if path.exists():
        intermediates[rank] = load_gemma3_trajectory(path)

print(f"r=4: {len(r4_trajs)} seeds")
print(f"r=16: {len(r16_trajs)} seeds")
print(f"Intermediates: {list(intermediates.keys())}")

# --- Build figure ---
fig = go.Figure()

# r=4 individual seeds (thin, transparent)
for i, (gens, ret) in enumerate(r4_trajs):
    fig.add_trace(go.Scatter(
        x=gens, y=ret, mode="lines",
        line=dict(color=C_HOMEO, width=1, dash="dot"),
        opacity=0.3, showlegend=False, hoverinfo="skip"
    ))

# r=4 mean (bold)
if r4_trajs:
    max_gen = min(len(t[0]) for t in r4_trajs)
    mean_r4 = [np.mean([r4_trajs[s][1][i] for s in range(len(r4_trajs))]) for i in range(max_gen)]
    gens_r4 = r4_trajs[0][0][:max_gen]
    fig.add_trace(go.Scatter(
        x=gens_r4, y=mean_r4, mode="lines+markers",
        name=f"r=4 (N={len(r4_trajs)}, homeostatic)",
        line=dict(color=C_HOMEO, width=3),
        marker=dict(size=7, symbol="circle"),
    ))

# Intermediates
int_styles = {
    10: dict(color=C_DEG_LIGHT, dash="dash", symbol="square"),
    12: dict(color=C_DEG_MED, dash="dash", symbol="diamond"),
    14: dict(color=C_DEG_DARK, dash="dash", symbol="triangle-up"),
}
for rank, (gens, ret) in intermediates.items():
    st = int_styles[rank]
    fig.add_trace(go.Scatter(
        x=gens, y=ret, mode="lines+markers",
        name=f"r={rank} (degradative)",
        line=dict(color=st["color"], width=2, dash=st["dash"]),
        marker=dict(size=6, symbol=st["symbol"]),
    ))

# r=16 individual seeds (thin, transparent)
for i, (gens, ret) in enumerate(r16_trajs):
    fig.add_trace(go.Scatter(
        x=gens, y=ret, mode="lines",
        line=dict(color=C_DEG_DEEP, width=1, dash="dot"),
        opacity=0.3, showlegend=False, hoverinfo="skip"
    ))

# r=16 mean (bold)
if r16_trajs:
    max_gen = min(len(t[0]) for t in r16_trajs)
    mean_r16 = [np.mean([r16_trajs[s][1][i] for s in range(len(r16_trajs))]) for i in range(max_gen)]
    gens_r16 = r16_trajs[0][0][:max_gen]
    fig.add_trace(go.Scatter(
        x=gens_r16, y=mean_r16, mode="lines+markers",
        name=f"r=16 (N={len(r16_trajs)}, degradative)",
        line=dict(color=C_DEG_DEEP, width=3),
        marker=dict(size=7, symbol="diamond"),
    ))

# 90% reference line
fig.add_hline(y=90, line_dash="dot", line_color="gray", opacity=0.5,
              annotation_text="90%", annotation_position="right")

# Layout
fig.update_layout(
    **LAYOUT_DEFAULTS,
    width=600, height=400,
    xaxis_title="Recursive Generation",
    yaxis_title="K₀ Retention (%)",
    yaxis_range=[55, 102],
    xaxis_range=[0.5, 5.5],
    xaxis_dtick=1,
    legend=dict(x=0.5, y=-0.18, xanchor="center", yanchor="top",
                orientation="h", bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=10)),
    margin=dict(l=60, r=30, t=15, b=70),
)
style_axes(fig)

# Save
pio.write_image(fig, str(FIG_DIR / "fig_gemma3_trajectories.png"), width=600, height=400, scale=4)
pio.write_image(fig, str(FIG_DIR / "fig_gemma3_trajectories.pdf"), width=600, height=400)
print(f"\nSaved: fig_gemma3_trajectories.png ({(FIG_DIR / 'fig_gemma3_trajectories.png').stat().st_size/1024:.0f} KB)")
print(f"Saved: fig_gemma3_trajectories.pdf")
