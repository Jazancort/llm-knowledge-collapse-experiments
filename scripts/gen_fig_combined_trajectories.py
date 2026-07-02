"""Generate combined Qwen + Gemma 3 trajectory figure (side-by-side subplots)."""
import json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

OUT = Path(__file__).parent.parent / "outputs"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figs"
FIG_DIR.mkdir(exist_ok=True)

FONT = "Inter, Arial, sans-serif"

# Colors
C_HOMEO = "#1a9641"
C_BOUNDED = "#fdae61"
C_DEG = "#d7191c"
C_DEG_DEEP = "#4A148C"
C_INTERMEDIATE = "#e66101"


def load_fft_gen10(filename, k0=78):
    data = json.loads((OUT / "fft_drift_gen10" / filename).read_text())
    gens = [0] + [r["gen"] for r in data]
    ret = [100.0] + [r["retention"] / k0 * 100 for r in data]
    return gens, ret


def load_g1_retention(folder, k0_override=None):
    data = json.loads((OUT / folder / "results.json").read_text())
    k0 = k0_override or data[0]["k0_size"]
    gens = [r["generation"] for r in data]
    ret = [r["retention"] / k0 * 100 for r in data]
    return gens, ret


def load_gemma3(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and 'generations' in data:
        k0 = data.get('k0_size', 47)
        gens = [g['gen'] for g in data['generations']]
        ret = [g['retention'] / k0 * 100 for g in data['generations']]
        return gens, ret
    elif isinstance(data, list) and data and 'generation' in data[0]:
        k0 = data[0].get('k0_size', 47)
        gens = [g['generation'] for g in data if g.get('generation', 0) > 0]
        ret = [g['retention'] / k0 * 100 for g in data if g.get('generation', 0) > 0]
        return gens, ret
    return [], []


# === QWEN DATA ===
# r=16 (3 seeds from fft_drift_gen10)
qwen_r16 = [load_fft_gen10(f"qlora_r16_seed{s}.json") for s in [15, 137, 256]]
# r=128 (1 seed)
qwen_r128 = load_g1_retention("g1_rank128_seed15")
# r=256 (3 seeds)
qwen_r256 = [load_g1_retention(f"g1_rank256_seed{s}") for s in [15, 137, 256]]

# === GEMMA 3 DATA ===
# r=4 (5 seeds)
gemma_r4 = []
for seed in [15, 137, 256]:
    path = OUT / f'gemma3_rank4_seed{seed}' / 'results.json'
    if path.exists():
        gemma_r4.append(load_gemma3(path))
for seed in [42, 77]:
    path = OUT / 'gemma3_extra_seeds' / f'gemma3_rank4_seed{seed}.json'
    if path.exists():
        gemma_r4.append(load_gemma3(path))

# r=16 (5 seeds)
gemma_r16 = []
for seed in [15, 137, 256]:
    path = OUT / f'gemma3_rank16_seed{seed}' / 'results.json'
    if path.exists():
        gemma_r16.append(load_gemma3(path))
for seed in [42, 77]:
    path = OUT / 'gemma3_extra_seeds' / f'gemma3_rank16_seed{seed}.json'
    if path.exists():
        gemma_r16.append(load_gemma3(path))

# Intermediates
gemma_int = {}
for rank in [10, 12, 14]:
    path = OUT / 'gemma3_intermediate_ranks' / f'gemma3_rank{rank}_seed15.json'
    if path.exists():
        gemma_int[rank] = load_gemma3(path)


# === BUILD FIGURE ===
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=["<b>(a)</b> Qwen 2.5 1.5B", "<b>(b)</b> Gemma 3 1B"],
    vertical_spacing=0.18,
)

# --- QWEN (left) ---
# r=16 seeds (thin)
for gens, ret in qwen_r16:
    fig.add_trace(go.Scatter(x=gens, y=ret, mode="lines",
        line=dict(color=C_HOMEO, width=1, dash="dot"), opacity=0.3,
        showlegend=False, hoverinfo="skip"), row=1, col=1)
# r=16 mean
mean_r16 = [np.mean([qwen_r16[s][1][i] for s in range(3)]) for i in range(len(qwen_r16[0][1]))]
fig.add_trace(go.Scatter(x=qwen_r16[0][0], y=mean_r16, mode="lines+markers",
    name="r=16 (homeostatic)", line=dict(color=C_HOMEO, width=3),
    marker=dict(size=5), legendgroup="homeo"), row=1, col=1)

# r=128
fig.add_trace(go.Scatter(x=qwen_r128[0], y=qwen_r128[1], mode="lines+markers",
    name="r=128 (bounded)", line=dict(color=C_BOUNDED, width=2.5),
    marker=dict(size=5, symbol="square"), legendgroup="bounded"), row=1, col=1)

# r=256 seeds (thin)
for gens, ret in qwen_r256:
    fig.add_trace(go.Scatter(x=gens, y=ret, mode="lines",
        line=dict(color=C_DEG, width=1, dash="dot"), opacity=0.3,
        showlegend=False, hoverinfo="skip"), row=1, col=1)
# r=256 mean
mean_r256 = [np.mean([qwen_r256[s][1][i] for s in range(3)]) for i in range(len(qwen_r256[0][1]))]
fig.add_trace(go.Scatter(x=qwen_r256[0][0], y=mean_r256, mode="lines+markers",
    name="r=256 (degradative)", line=dict(color=C_DEG, width=3),
    marker=dict(size=5, symbol="diamond"), legendgroup="degrad"), row=1, col=1)

# --- GEMMA 3 (bottom) ---
# r=4 seeds (thin)
for gens, ret in gemma_r4:
    fig.add_trace(go.Scatter(x=gens, y=ret, mode="lines",
        line=dict(color=C_HOMEO, width=1, dash="dot"), opacity=0.3,
        showlegend=False, hoverinfo="skip"), row=2, col=1)
# r=4 mean
max_gen = min(len(t[0]) for t in gemma_r4)
mean_g4 = [np.mean([gemma_r4[s][1][i] for s in range(len(gemma_r4))]) for i in range(max_gen)]
fig.add_trace(go.Scatter(x=gemma_r4[0][0][:max_gen], y=mean_g4, mode="lines+markers",
    name="r=4 (homeostatic)", line=dict(color=C_HOMEO, width=3),
    marker=dict(size=5), legendgroup="homeo", showlegend=False), row=2, col=1)

# Intermediates
for rank in [10, 14]:
    if rank in gemma_int:
        gens, ret = gemma_int[rank]
        fig.add_trace(go.Scatter(x=gens, y=ret, mode="lines+markers",
            name=f"r={rank} (degradative)", line=dict(color=C_INTERMEDIATE, width=2, dash="dash"),
            marker=dict(size=5, symbol="triangle-up"), legendgroup="int",
            showlegend=(rank == 10)), row=2, col=1)

# r=16 seeds (thin)
for gens, ret in gemma_r16:
    fig.add_trace(go.Scatter(x=gens, y=ret, mode="lines",
        line=dict(color=C_DEG_DEEP, width=1, dash="dot"), opacity=0.3,
        showlegend=False, hoverinfo="skip"), row=2, col=1)
# r=16 mean
max_gen = min(len(t[0]) for t in gemma_r16)
mean_g16 = [np.mean([gemma_r16[s][1][i] for s in range(len(gemma_r16))]) for i in range(max_gen)]
fig.add_trace(go.Scatter(x=gemma_r16[0][0][:max_gen], y=mean_g16, mode="lines+markers",
    name="r=16 (degradative)", line=dict(color=C_DEG_DEEP, width=3),
    marker=dict(size=5, symbol="diamond"), legendgroup="degrad", showlegend=False), row=2, col=1)

# 90% reference lines
fig.add_hline(y=90, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=1)
fig.add_hline(y=90, line_dash="dot", line_color="gray", opacity=0.4, row=2, col=1)

# Layout
fig.update_xaxes(title_text="Generation", row=1, col=1, range=[-0.5, 10.5], dtick=2,
                 showgrid=False, linecolor="#ccc", linewidth=1, mirror=True)
fig.update_xaxes(title_text="Generation", row=2, col=1, range=[0.5, 5.5], dtick=1,
                 showgrid=False, linecolor="#ccc", linewidth=1, mirror=True)
fig.update_yaxes(title_text="K₀ Retention (%)", row=1, col=1, range=[60, 102],
                 showgrid=True, gridcolor="#eee", linecolor="#ccc", linewidth=1, mirror=True)
fig.update_yaxes(title_text="K₀ Retention (%)", row=2, col=1, range=[60, 102],
                 showgrid=True, gridcolor="#eee", linecolor="#ccc", linewidth=1, mirror=True)

fig.update_layout(
    font=dict(family=FONT, size=13),
    paper_bgcolor="white",
    plot_bgcolor="white",
    width=480, height=620,
    legend=dict(x=0.5, y=-0.16, xanchor="center", yanchor="top",
                orientation="h", font=dict(size=12)),
    margin=dict(l=55, r=15, t=30, b=60),
)

# Save
pio.write_image(fig, str(FIG_DIR / "fig_combined_trajectories.png"), width=480, height=620, scale=4)
pio.write_image(fig, str(FIG_DIR / "fig_combined_trajectories.pdf"), width=480, height=620)
sz = (FIG_DIR / "fig_combined_trajectories.png").stat().st_size / 1024
print(f"Saved: fig_combined_trajectories.png ({sz:.0f} KB)")
print(f"Saved: fig_combined_trajectories.pdf")
