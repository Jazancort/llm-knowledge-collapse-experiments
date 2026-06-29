"""
Figure 1: Longitudinal K0 Retention Trajectories
Qwen 2.5 1.5B, r=16 (N=3), r=128 (N=1), r=256 (N=3), Gen0-10
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


def load_g1_retention(folder):
    """Load from g1_rank* format: list of dicts with 'generation' and 'retention'."""
    data = json.loads((OUT / folder / "results.json").read_text())
    k0 = data[0]["k0_size"]
    gens = [r["generation"] for r in data]
    ret = [r["retention"] / k0 * 100 for r in data]
    return gens, ret


def load_fft_gen10(filename, k0=78):
    """Load from fft_drift_gen10 format: list starting at gen1 with 'retention' count."""
    data = json.loads((OUT / "fft_drift_gen10" / filename).read_text())
    gens = [0] + [r["gen"] for r in data]
    ret = [100.0] + [r["retention"] / k0 * 100 for r in data]
    return gens, ret


# r=16 (N=3) from fft_drift_gen10 (QLoRA r=16, 10 gens, 3 seeds)
r16_seeds = {}
for seed in [15, 137, 256]:
    r16_seeds[seed] = load_fft_gen10(f"qlora_r16_seed{seed}.json")

# r=128 (N=1) from g1_rank128_seed15
r128_gens, r128_ret = load_g1_retention("g1_rank128_seed15")

# r=256 (N=3) from g1_rank256_seed*
r256_seeds = {}
for seed in [15, 137, 256]:
    r256_seeds[seed] = load_g1_retention(f"g1_rank256_seed{seed}")

# Plot
fig, ax = plt.subplots(figsize=(3.3, 2.4))

# r=16 (blue) - thin per-seed, bold mean
for seed, (g, r) in r16_seeds.items():
    ax.plot(g, r, color="C0", alpha=0.35, linewidth=0.7, zorder=2)
r16_mean = np.mean([r16_seeds[s][1] for s in r16_seeds], axis=0)
ax.plot(r16_seeds[15][0], r16_mean, color="C0", linewidth=1.8,
        label="$r=16$ (N=3)", zorder=3)

# r=128 (orange) - single seed, dashed
ax.plot(r128_gens, r128_ret, color="C1", linewidth=1.8, linestyle="--",
        label="$r=128$ (N=1)", zorder=3)

# r=256 (red) - thin per-seed, bold mean
for seed, (g, r) in r256_seeds.items():
    ax.plot(g, r, color="C3", alpha=0.35, linewidth=0.7, zorder=2)
r256_mean = np.mean([r256_seeds[s][1] for s in r256_seeds], axis=0)
ax.plot(r256_seeds[15][0], r256_mean, color="C3", linewidth=1.8,
        label="$r=256$ (N=3)", zorder=3)

# Formatting
ax.set_xlabel("Generation")
ax.set_ylabel("K0 Retention (%)")
ax.set_xlim(0, 10)
ax.set_ylim(60, 102)
ax.set_xticks(range(0, 11, 2))
ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
ax.grid(True, alpha=0.3)

# Save
fig.savefig(FIG_DIR / "fig1_trajectories.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "fig1_trajectories.svg", bbox_inches="tight")
print("Saved fig1_trajectories.pdf and .svg")
plt.close()
