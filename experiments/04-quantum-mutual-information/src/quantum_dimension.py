"""
Dimension extraction for quantum mutual information experiment.

Uses MDS and Isomap on MI-derived distance matrices to extract
effective quantum dimension d_Q.
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.manifold import MDS, Isomap
from sklearn.metrics import pairwise_distances
import warnings


def mi_distance_matrix(MI: np.ndarray, method: str = 'sqrt') -> np.ndarray:
    """
    Convert mutual information matrix to distance matrix.

    Args:
        MI: Mutual information matrix (N x N)
        method: Conversion method
            - 'sqrt': D = sqrt(2 * (S_max - MI))
            - 'inverse': D = 1 / (MI + epsilon)
            - 'negative': D = S_max - MI

    Returns:
        Distance matrix
    """
    n = MI.shape[0]
    S_max = 1.0  # Max entropy for single qubit

    if method == 'sqrt':
        D = np.sqrt(2 * np.maximum(0, S_max - MI))
    elif method == 'inverse':
        epsilon = 0.01
        D = 1.0 / (MI + epsilon)
    elif method == 'negative':
        D = S_max - MI
    else:
        raise ValueError(f"Unknown method: {method}")

    np.fill_diagonal(D, 0)
    return D


def extract_dimension_mds(D: np.ndarray, max_dim: int = 10,
                          threshold: float = 0.9) -> Tuple[float, dict]:
    """
    Extract effective dimension using MDS.

    Uses explained variance to determine dimension where
    reconstruction quality exceeds threshold.

    Args:
        D: Distance matrix
        max_dim: Maximum dimension to test
        threshold: Explained variance threshold

    Returns:
        Tuple of (effective dimension, details dict)
    """
    n = D.shape[0]
    max_dim = min(max_dim, n - 1)

    # Check for degenerate distance matrix (all zeros or near-constant)
    D_offdiag = D[np.triu_indices(n, k=1)]
    if len(D_offdiag) == 0 or np.std(D_offdiag) < 1e-10:
        # Degenerate case: all distances equal or zero
        return 1.0, {
            'explained_variances': [1.0],
            'd_threshold': 1,
            'd_continuous': 1.0,
            'final_variance': 1.0,
            'degenerate': True
        }

    explained_variances = []
    embeddings = {}

    for d in range(1, max_dim + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mds = MDS(n_components=d, dissimilarity='precomputed',
                      random_state=42, n_init=4, max_iter=300,
                      normalized_stress='auto')
            try:
                embedding = mds.fit_transform(D)
                embeddings[d] = embedding

                # Compute reconstruction quality
                D_reconstructed = pairwise_distances(embedding)

                # Explained variance: 1 - SS_residual / SS_total
                ss_total = np.sum(D ** 2)
                ss_residual = np.sum((D - D_reconstructed) ** 2)

                if ss_total > 1e-10:
                    explained = max(0, min(1, 1 - ss_residual / ss_total))
                else:
                    explained = 1.0

                explained_variances.append(explained)
            except Exception:
                explained_variances.append(0.0)

    # Find dimension where threshold is exceeded
    d_effective = max_dim
    for i, ev in enumerate(explained_variances):
        if ev >= threshold:
            d_effective = i + 1
            break

    # Compute continuous dimension (fractional based on interpolation)
    d_continuous = float(d_effective)
    if d_effective > 1 and d_effective <= len(explained_variances):
        # Linear interpolation between d-1 and d
        ev_below = explained_variances[d_effective - 2] if d_effective > 1 else 0
        ev_at = explained_variances[d_effective - 1]
        if ev_at > ev_below:
            frac = (threshold - ev_below) / (ev_at - ev_below)
            d_continuous = (d_effective - 1) + frac

    return d_continuous, {
        'explained_variances': explained_variances,
        'd_threshold': d_effective,
        'd_continuous': d_continuous,
        'final_variance': explained_variances[-1] if explained_variances else 0
    }


def extract_dimension_isomap(D: np.ndarray, max_dim: int = 10,
                              threshold: float = 0.9,
                              n_neighbors: int = 5) -> Tuple[float, dict]:
    """
    Extract effective dimension using Isomap.

    Args:
        D: Distance matrix
        max_dim: Maximum dimension to test
        threshold: Explained variance threshold
        n_neighbors: Number of neighbors for Isomap

    Returns:
        Tuple of (effective dimension, details dict)
    """
    n = D.shape[0]
    max_dim = min(max_dim, n - 2)
    n_neighbors = min(n_neighbors, n - 1)

    # Check for degenerate distance matrix
    D_offdiag = D[np.triu_indices(n, k=1)]
    if len(D_offdiag) == 0 or np.std(D_offdiag) < 1e-10:
        return 1.0, {
            'explained_variances': [1.0],
            'd_threshold': 1,
            'd_continuous': 1.0,
            'final_variance': 1.0,
            'degenerate': True
        }

    explained_variances = []

    for d in range(1, max_dim + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                isomap = Isomap(n_components=d, n_neighbors=n_neighbors,
                                metric='precomputed')
                embedding = isomap.fit_transform(D)

                # Compute reconstruction quality
                D_reconstructed = pairwise_distances(embedding)

                ss_total = np.sum(D ** 2)
                ss_residual = np.sum((D - D_reconstructed) ** 2)

                if ss_total > 1e-10:
                    explained = max(0, min(1, 1 - ss_residual / ss_total))
                else:
                    explained = 1.0

                explained_variances.append(explained)
            except Exception:
                explained_variances.append(0.0)

    # Find dimension where threshold is exceeded
    d_effective = max_dim
    for i, ev in enumerate(explained_variances):
        if ev >= threshold:
            d_effective = i + 1
            break

    # Continuous dimension
    d_continuous = float(d_effective)
    if d_effective > 1 and d_effective <= len(explained_variances):
        ev_below = explained_variances[d_effective - 2] if d_effective > 1 else 0
        ev_at = explained_variances[d_effective - 1]
        if ev_at > ev_below:
            frac = (threshold - ev_below) / (ev_at - ev_below)
            d_continuous = (d_effective - 1) + frac

    return d_continuous, {
        'explained_variances': explained_variances,
        'd_threshold': d_effective,
        'd_continuous': d_continuous,
        'final_variance': explained_variances[-1] if explained_variances else 0
    }


def compute_compression(d_topo: float, d_quantum: float) -> float:
    """
    Compute compression ratio.

    delta = (d_topo - d_Q) / d_topo

    Args:
        d_topo: Topological dimension (N-1 for N qubits in chain)
        d_quantum: Quantum dimension from MI geometry

    Returns:
        Compression ratio
    """
    if d_topo <= 0:
        return 0.0
    return (d_topo - d_quantum) / d_topo


def analyze_quantum_state(MI: np.ndarray, n_qubits: int,
                          geometry: str = 'chain') -> dict:
    """
    Full analysis of quantum state from its MI matrix.

    Args:
        MI: Mutual information matrix
        n_qubits: Number of qubits
        geometry: 'chain' or '2x4', '3x4', etc.

    Returns:
        Analysis results dictionary
    """
    # Convert MI to distance
    D = mi_distance_matrix(MI, method='sqrt')

    # Topological dimension
    if geometry == 'chain':
        d_topo = n_qubits - 1  # 1D chain has d = N-1 in embedding
    elif 'x' in geometry:
        rows, cols = map(int, geometry.split('x'))
        d_topo = min(rows, cols) * 2 - 1  # Approximate for 2D grid
    else:
        d_topo = n_qubits - 1

    # Extract quantum dimensions
    d_mds, mds_details = extract_dimension_mds(D)
    d_isomap, isomap_details = extract_dimension_isomap(D)

    # Average dimension
    d_quantum = (d_mds + d_isomap) / 2

    # Compression
    delta = compute_compression(d_topo, d_quantum)

    return {
        'n_qubits': n_qubits,
        'geometry': geometry,
        'd_topo': d_topo,
        'd_mds': d_mds,
        'd_isomap': d_isomap,
        'd_quantum': d_quantum,
        'delta': delta,
        'mi_mean': float(np.mean(MI[np.triu_indices(n_qubits, k=1)])),
        'mi_max': float(np.max(MI)),
        'mi_min': float(np.min(MI[np.triu_indices(n_qubits, k=1)])),
        'mds_details': mds_details,
        'isomap_details': isomap_details
    }


def compare_state_dimensions(results: List[dict]) -> dict:
    """
    Compare dimensional analysis across multiple quantum states.

    Args:
        results: List of analysis results from analyze_quantum_state

    Returns:
        Comparison summary
    """
    by_state = {}
    for r in results:
        state = r.get('state_name', 'unknown')
        if state not in by_state:
            by_state[state] = []
        by_state[state].append(r)

    summary = {}
    for state, state_results in by_state.items():
        deltas = [r['delta'] for r in state_results]
        d_quantums = [r['d_quantum'] for r in state_results]

        summary[state] = {
            'mean_delta': float(np.mean(deltas)),
            'std_delta': float(np.std(deltas)),
            'mean_d_quantum': float(np.mean(d_quantums)),
            'n_samples': len(state_results)
        }

    return summary
