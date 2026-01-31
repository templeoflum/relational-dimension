"""
Consistent dimension estimation methods for Experiment 03.

Uses ONLY full methods (no sparse approximations) to ensure consistency
across all graph sizes from N=50 to N=1000.
"""

import numpy as np
from sklearn.manifold import Isomap, MDS
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Dict, Any, Optional
import warnings


def full_mds_dimension(D: np.ndarray, k_max: int = 10,
                       threshold: float = 0.1) -> Dict[str, Any]:
    """
    Extract intrinsic dimension using full classical MDS.

    No approximations - computes full distance matrix embedding.

    Args:
        D: N x N distance matrix
        k_max: Maximum dimension to test
        threshold: Error threshold for dimension detection

    Returns:
        Dictionary with dimension estimate, errors, embedding, and stress
    """
    n = D.shape[0]
    k_max = min(k_max, n - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mds = MDS(n_components=k_max, dissimilarity='precomputed',
                  normalized_stress='auto', random_state=42, max_iter=500)
        try:
            embedding = mds.fit_transform(D)
            stress = mds.stress_
        except Exception as e:
            return {
                'd_mds': k_max,
                'errors': np.ones(k_max),
                'embedding': np.zeros((n, k_max)),
                'stress': 1.0,
                'valid': False,
                'error_msg': str(e)
            }

    # Compute reconstruction errors for each dimension
    errors = compute_reconstruction_errors(D, embedding, k_max)

    # Find intrinsic dimension with interpolation for fractional estimate
    d_estimate, d_fractional = find_dimension_with_interpolation(errors, threshold)

    return {
        'd_mds': d_estimate,
        'd_fractional': d_fractional,
        'errors': errors,
        'embedding': embedding,
        'stress': float(stress) if stress is not None else 0.0,
        'valid': True
    }


def full_isomap_dimension(D: np.ndarray, k_max: int = 10,
                          n_neighbors: int = 8,
                          threshold: float = 0.1) -> Dict[str, Any]:
    """
    Extract intrinsic dimension using full Isomap.

    No sparse approximations - uses full geodesic distance matrix.

    Args:
        D: N x N distance matrix (precomputed)
        k_max: Maximum dimension to test
        n_neighbors: Number of neighbors for Isomap graph
        threshold: Error threshold for dimension detection

    Returns:
        Dictionary with dimension estimate, errors, embedding
    """
    n = D.shape[0]
    n_neighbors = min(n_neighbors, n - 1)
    k_max = min(k_max, n - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isomap = Isomap(n_components=k_max, n_neighbors=n_neighbors,
                        metric='precomputed')
        try:
            embedding = isomap.fit_transform(D)
        except Exception as e:
            return {
                'd_isomap': k_max,
                'errors': np.ones(k_max),
                'embedding': np.zeros((n, k_max)),
                'valid': False,
                'error_msg': str(e)
            }

    # Compute reconstruction errors
    errors = compute_reconstruction_errors(D, embedding, k_max)

    # Find intrinsic dimension
    d_estimate, d_fractional = find_dimension_with_interpolation(errors, threshold)

    return {
        'd_isomap': d_estimate,
        'd_fractional': d_fractional,
        'errors': errors,
        'embedding': embedding,
        'valid': True
    }


def compute_reconstruction_errors(D_original: np.ndarray, embedding: np.ndarray,
                                   k_max: int) -> np.ndarray:
    """
    Compute reconstruction error for each embedding dimension.

    Error is normalized RMSE between original and reconstructed distances.

    Args:
        D_original: Original distance matrix
        embedding: Full k_max dimensional embedding
        k_max: Maximum dimension to compute

    Returns:
        Array of errors for dimensions 1..k_max
    """
    errors = np.zeros(k_max)

    n = D_original.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    D_orig_flat = D_original[triu_idx]

    D_max = np.max(D_orig_flat)
    if D_max == 0:
        return np.zeros(k_max)

    for k in range(1, k_max + 1):
        emb_k = embedding[:, :k]
        D_recon = squareform(pdist(emb_k))
        D_recon_flat = D_recon[triu_idx]

        # Normalize reconstructed distances
        recon_max = np.max(D_recon_flat)
        if recon_max > 0:
            D_recon_flat = D_recon_flat * (D_max / recon_max)

        # Normalized RMSE
        errors[k - 1] = np.sqrt(np.mean((D_orig_flat - D_recon_flat) ** 2)) / D_max

    return errors


def find_dimension_with_interpolation(errors: np.ndarray,
                                       threshold: float = 0.1) -> Tuple[int, float]:
    """
    Find intrinsic dimension with linear interpolation for fractional estimate.

    Args:
        errors: Array of reconstruction errors for each dimension
        threshold: Fraction of initial error to use as cutoff

    Returns:
        Tuple of (integer dimension, fractional dimension)
    """
    if len(errors) == 0:
        return 1, 1.0

    initial_error = errors[0]
    if initial_error == 0:
        return 1, 1.0

    cutoff = threshold * initial_error

    # Find first dimension below threshold
    below_threshold = np.where(errors < cutoff)[0]

    if len(below_threshold) > 0:
        d_int = below_threshold[0] + 1

        # Linear interpolation for fractional dimension
        if d_int > 1:
            e_prev = errors[d_int - 2]  # Error at d_int - 1
            e_curr = errors[d_int - 1]  # Error at d_int
            if e_prev != e_curr:
                # Interpolate: find where error = cutoff between d_int-1 and d_int
                frac = (e_prev - cutoff) / (e_prev - e_curr)
                d_frac = (d_int - 1) + frac
            else:
                d_frac = float(d_int)
        else:
            d_frac = float(d_int)

        return d_int, d_frac
    else:
        # Use elbow method
        d_int = find_elbow(errors)
        return d_int, float(d_int)


def find_elbow(errors: np.ndarray) -> int:
    """
    Find elbow point in error curve using curvature.
    """
    if len(errors) < 3:
        return len(errors)

    d2 = np.diff(np.diff(errors))

    if len(d2) == 0:
        return 1

    elbow_idx = np.argmax(d2) + 1
    return max(1, min(elbow_idx + 1, len(errors)))


def extract_dimension_consistent(D: np.ndarray, k_max: int = 10,
                                  n_neighbors: int = 8,
                                  threshold: float = 0.1) -> Dict[str, Any]:
    """
    Extract dimension using both MDS and Isomap with validation.

    This is the main function for Experiment 03 - ensures both methods
    are applied consistently without approximations.

    Args:
        D: Distance matrix
        k_max: Maximum dimension to test
        n_neighbors: Number of neighbors for Isomap
        threshold: Error threshold for dimension detection

    Returns:
        Dictionary with:
        - d_mds: MDS dimension estimate
        - d_isomap: Isomap dimension estimate
        - d_mean: Average dimension
        - d_fractional_mean: Average fractional dimension
        - valid: Whether methods agree within tolerance
        - agreement_diff: Difference between methods
        - stress: MDS stress value
    """
    mds_result = full_mds_dimension(D, k_max, threshold)
    isomap_result = full_isomap_dimension(D, k_max, n_neighbors, threshold)

    d_mds = mds_result.get('d_mds', k_max)
    d_isomap = isomap_result.get('d_isomap', k_max)

    d_frac_mds = mds_result.get('d_fractional', float(d_mds))
    d_frac_isomap = isomap_result.get('d_fractional', float(d_isomap))

    diff = abs(d_mds - d_isomap)
    valid = diff <= 0.5 and mds_result.get('valid', False) and isomap_result.get('valid', False)

    return {
        'd_mds': d_mds,
        'd_isomap': d_isomap,
        'd_mean': (d_mds + d_isomap) / 2,
        'd_fractional_mds': d_frac_mds,
        'd_fractional_isomap': d_frac_isomap,
        'd_fractional_mean': (d_frac_mds + d_frac_isomap) / 2,
        'valid': valid,
        'agreement_diff': diff,
        'stress': mds_result.get('stress', 0.0),
        'errors_mds': mds_result.get('errors', np.array([])),
        'errors_isomap': isomap_result.get('errors', np.array([])),
        'mds_valid': mds_result.get('valid', False),
        'isomap_valid': isomap_result.get('valid', False)
    }


def validate_embedding_quality(stress: float, agreement_diff: float,
                                max_stress: float = 0.2,
                                max_diff: float = 0.5) -> Dict[str, Any]:
    """
    Validate the quality of dimension extraction.

    Args:
        stress: MDS stress value
        agreement_diff: Difference between MDS and Isomap dimensions
        max_stress: Maximum acceptable stress
        max_diff: Maximum acceptable method disagreement

    Returns:
        Validation result dictionary
    """
    stress_ok = stress < max_stress
    agreement_ok = agreement_diff <= max_diff

    return {
        'stress_ok': stress_ok,
        'agreement_ok': agreement_ok,
        'overall_valid': stress_ok and agreement_ok,
        'stress': stress,
        'agreement_diff': agreement_diff
    }


def compute_compression_ratio(d_topo: float, d_corr: float) -> float:
    """
    Compute compression ratio delta.

    delta = (d_topo - d_corr) / d_topo

    Args:
        d_topo: Topological dimension
        d_corr: Correlation dimension

    Returns:
        Compression ratio (positive = compression)
    """
    if d_topo == 0:
        return 0.0
    return (d_topo - d_corr) / d_topo
