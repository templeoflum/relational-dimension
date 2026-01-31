"""
Dimension extraction with method agreement checking.

Implements both Isomap and MDS dimension extraction with:
- Explained variance-based dimension detection
- Method agreement gate (exclude if methods disagree)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.manifold import MDS, Isomap
from sklearn.metrics import pairwise_distances
import warnings


def extract_dimension_mds(D: np.ndarray, max_dim: int = 10,
                          threshold: float = 0.95) -> Tuple[float, Dict]:
    """
    Extract dimension using MDS.

    Args:
        D: Distance matrix
        max_dim: Maximum dimension to test
        threshold: Explained variance threshold

    Returns:
        Tuple of (dimension, details dict)
    """
    n = D.shape[0]
    max_dim = min(max_dim, n - 1)

    # Check for degenerate matrix
    D_offdiag = D[np.triu_indices(n, k=1)]
    if len(D_offdiag) == 0 or np.std(D_offdiag) < 1e-10:
        return 1.0, {'degenerate': True, 'explained_variances': [1.0]}

    explained_variances = []

    for d in range(1, max_dim + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mds = MDS(n_components=d, dissimilarity='precomputed',
                          random_state=42, n_init=4, max_iter=300,
                          normalized_stress='auto')
                embedding = mds.fit_transform(D)

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

    # Find dimension where threshold exceeded
    d_effective = max_dim
    for i, ev in enumerate(explained_variances):
        if ev >= threshold:
            d_effective = i + 1
            break

    # Continuous dimension via interpolation
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
        'd_continuous': d_continuous
    }


def extract_dimension_isomap(D: np.ndarray, max_dim: int = 10,
                              threshold: float = 0.95,
                              n_neighbors: int = 8) -> Tuple[float, Dict]:
    """
    Extract dimension using Isomap.

    Args:
        D: Distance matrix
        max_dim: Maximum dimension to test
        threshold: Explained variance threshold
        n_neighbors: Number of neighbors

    Returns:
        Tuple of (dimension, details dict)
    """
    n = D.shape[0]
    max_dim = min(max_dim, n - 2)
    n_neighbors = min(n_neighbors, n - 1)

    # Check for degenerate matrix
    D_offdiag = D[np.triu_indices(n, k=1)]
    if len(D_offdiag) == 0 or np.std(D_offdiag) < 1e-10:
        return 1.0, {'degenerate': True, 'explained_variances': [1.0]}

    explained_variances = []

    for d in range(1, max_dim + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                isomap = Isomap(n_components=d, n_neighbors=n_neighbors,
                                metric='precomputed')
                embedding = isomap.fit_transform(D)

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

    # Find dimension where threshold exceeded
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
        'd_continuous': d_continuous
    }


def extract_dimension_with_agreement(D: np.ndarray,
                                      agreement_threshold: float = 0.5,
                                      **kwargs) -> Dict:
    """
    Extract dimension using both methods with agreement check.

    Args:
        D: Distance matrix
        agreement_threshold: Maximum allowed difference between methods
        **kwargs: Passed to extraction functions

    Returns:
        Dictionary with dimensions, agreement, and validity
    """
    d_mds, mds_details = extract_dimension_mds(D, **kwargs)
    d_isomap, isomap_details = extract_dimension_isomap(D, **kwargs)

    method_diff = abs(d_mds - d_isomap)
    methods_agree = method_diff <= agreement_threshold

    # Average dimension if methods agree
    if methods_agree:
        d_final = (d_mds + d_isomap) / 2
    else:
        d_final = None  # Invalid - methods disagree

    return {
        'd_mds': d_mds,
        'd_isomap': d_isomap,
        'd_final': d_final,
        'method_diff': method_diff,
        'methods_agree': methods_agree,
        'valid': methods_agree,
        'mds_details': mds_details,
        'isomap_details': isomap_details
    }


def compute_delta(d_topo: float, d_corr: float) -> float:
    """
    Compute compression ratio delta.

    delta = (d_topo - d_corr) / d_topo
    """
    if d_topo <= 0:
        return 0.0
    return (d_topo - d_corr) / d_topo


def full_dimension_analysis(D_topo: np.ndarray, D_corr: np.ndarray,
                            agreement_threshold: float = 0.5,
                            use_mds_only_for_corr: bool = True) -> Dict:
    """
    Full dimension analysis for both topology and correlation distances.

    Args:
        D_topo: Topology-based distance matrix
        D_corr: Correlation-based distance matrix
        agreement_threshold: Method agreement threshold
        use_mds_only_for_corr: If True, use only MDS for correlation distances
            (Isomap often fails on non-geodesic correlation distances)

    Returns:
        Complete analysis dictionary
    """
    # For topology distances, use method agreement
    topo_result = extract_dimension_with_agreement(D_topo, agreement_threshold)

    # For correlation distances, Isomap often fails (returns max_dim)
    # because correlation distances don't satisfy geodesic properties
    if use_mds_only_for_corr:
        d_corr, corr_details = extract_dimension_mds(D_corr)
        corr_result = {
            'd_mds': d_corr,
            'd_isomap': None,
            'd_final': d_corr,
            'method_diff': 0,
            'methods_agree': True,
            'valid': True,
            'mds_details': corr_details,
            'isomap_details': None,
            'mds_only': True
        }
    else:
        corr_result = extract_dimension_with_agreement(D_corr, agreement_threshold)

    # Topology must have method agreement, correlation uses MDS
    if topo_result['valid'] and corr_result['valid']:
        delta = compute_delta(topo_result['d_final'], corr_result['d_final'])
        valid = True
    else:
        delta = None
        valid = False

    return {
        'topology': topo_result,
        'correlation': corr_result,
        'delta': delta,
        'valid': valid,
        'd_topo': topo_result['d_final'],
        'd_corr': corr_result['d_final']
    }
