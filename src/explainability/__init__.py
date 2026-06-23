import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear CKA between two representation matrices.

    Args:
        X: (n_samples, n_features) from model A
        Y: (n_samples, n_features) from model B
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def compute_cka_by_blocks(hidden_states_a: list[np.ndarray],
                          hidden_states_b: list[np.ndarray],
                          layer_blocks: dict[str, list[int]]) -> dict[str, float]:
    """Compute CKA between two sets of hidden states, grouped by layer blocks.

    Args:
        hidden_states_a: list of (n_samples, hidden_dim) per layer
        hidden_states_b: list of (n_samples, hidden_dim) per layer
        layer_blocks: {"early": [0, 8], "middle": [8, 20], "late": [20, 32]}
    """
    results = {}
    num_layers = min(len(hidden_states_a), len(hidden_states_b))

    for block_name, (start, end) in layer_blocks.items():
        end = min(end, num_layers)
        if start >= num_layers:
            continue

        cka_values = []
        for i in range(start, end):
            cka = linear_cka(hidden_states_a[i], hidden_states_b[i])
            cka_values.append(cka)

        results[block_name] = float(np.mean(cka_values))

    return results


def extract_mean_pooled_states(hidden_states_tuple, n_samples: int) -> list[np.ndarray]:
    """Convert hidden states from multiple forward passes into mean-pooled arrays.

    Args:
        hidden_states_tuple: list of (n_layers,) tuples, each layer is (1, seq_len, hidden_dim)
        n_samples: number of samples processed

    Returns:
        list of (n_samples, hidden_dim) arrays, one per layer
    """
    n_layers = len(hidden_states_tuple[0])
    pooled = [[] for _ in range(n_layers)]

    for sample_states in hidden_states_tuple:
        for layer_idx in range(n_layers):
            state = sample_states[layer_idx]
            if hasattr(state, "numpy"):
                state = state.numpy()
            mean_vec = state.squeeze(0).mean(axis=0)
            pooled[layer_idx].append(mean_vec)

    return [np.stack(layer_vecs) for layer_vecs in pooled]
