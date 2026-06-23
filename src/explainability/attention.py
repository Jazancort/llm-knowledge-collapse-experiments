import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


def attention_rollout(attentions: list[np.ndarray]) -> np.ndarray:
    """Compute attention rollout across all layers.

    Args:
        attentions: list of (n_heads, seq_len, seq_len) per layer

    Returns:
        (seq_len,) aggregated attention to each input token from the last position
    """
    rollout = np.eye(attentions[0].shape[-1])

    for attn in attentions:
        attn_heads_mean = attn.mean(axis=0)
        attn_with_residual = 0.5 * attn_heads_mean + 0.5 * np.eye(attn_heads_mean.shape[0])
        rollout = attn_with_residual @ rollout

    return rollout[-1]


def compute_esi(rollout_a: np.ndarray, rollout_b: np.ndarray,
                alpha: float = 0.5, beta: float = 0.5) -> dict[str, float]:
    """Compute Explanation Stability Index between two attention rollouts.

    Args:
        rollout_a: (seq_len,) from generation t
        rollout_b: (seq_len,) from generation t+1 (same input, possibly different seq_len)

    Returns:
        dict with js_divergence, rank_correlation, and esi
    """
    min_len = min(len(rollout_a), len(rollout_b))
    a = rollout_a[:min_len]
    b = rollout_b[:min_len]

    a_norm = a / (a.sum() + 1e-10)
    b_norm = b / (b.sum() + 1e-10)

    js = float(jensenshannon(a_norm, b_norm) ** 2)
    rho, _ = spearmanr(a, b)
    rho = float(rho) if not np.isnan(rho) else 0.0

    esi = alpha * js + beta * (1.0 - rho)

    return {"js_divergence": js, "rank_correlation": rho, "esi": esi}


def batch_attention_rollout(all_attentions: list) -> list[np.ndarray]:
    """Compute attention rollout for a batch of samples.

    Args:
        all_attentions: list of per-sample attention tuples.
            Each is a tuple of (n_heads, seq_len, seq_len) per layer.
    """
    rollouts = []
    for sample_attn in all_attentions:
        layers = [a.squeeze(0).numpy() if hasattr(a, "numpy") else a.squeeze(0)
                  for a in sample_attn]
        rollouts.append(attention_rollout(layers))
    return rollouts
