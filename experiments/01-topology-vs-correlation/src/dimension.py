"""
Dimension extraction module for Experiment 01.

Provides functions to estimate intrinsic dimensionality from distance matrices
using Isomap and MDS embeddings.
"""

import numpy as np
from sklearn.manifold import Isomap, MDS
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional, List
import warnings


def extract_dimension_isomap(D: np.ndarray, k_max: int = 10,
                             n_neighbors: int = 8) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Extract intrinsic dimension using Isomap embedding.

    Args:
        D: N x N distance matrix (precomputed)
        k_max: Maximum dimension to test
        n_neighbors: Number of neighbors for Isomap

    Returns:
        Tuple of (estimated dimension, error curve, embedding)
    """
    n = D.shape[0]

    # Adjust n_neighbors if needed
    n_neighbors = min(n_neighbors, n - 1)
    k_max = min(k_max, n - 1)

    # Fit Isomap with maximum dimensions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isomap = Isomap(n_components=k_max, n_neighbors=n_neighbors,
                        metric='precomputed')
        try:
            embedding = isomap.fit_transform(D)
        except Exception as e:
            # If Isomap fails (e.g., disconnected graph), return max dimension
            return k_max, np.ones(k_max), np.zeros((n, k_max))

    # Compute reconstruction errors for each dimension
    errors = compute_reconstruction_errors(D, embedding, k_max)

    # Find intrinsic dimension
    d_estimate = find_intrinsic_dimension(errors)

    return d_estimate, errors, embedding


def extract_dimension_mds(D: np.ndarray, k_max: int = 10) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Extract intrinsic dimension using classical MDS.

    Args:
        D: N x N distance matrix
        k_max: Maximum dimension to test

    Returns:
        Tuple of (estimated dimension, error curve, embedding)
    """
    n = D.shape[0]
    k_max = min(k_max, n - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use classical MDS with precomputed dissimilarities
        mds = MDS(n_components=k_max, dissimilarity='precomputed',
                  normalized_stress='auto', random_state=42, max_iter=300)
        try:
            embedding = mds.fit_transform(D)
        except Exception:
            return k_max, np.ones(k_max), np.zeros((n, k_max))

    # Compute reconstruction errors
    errors = compute_reconstruction_errors(D, embedding, k_max)

    # Find intrinsic dimension
    d_estimate = find_intrinsic_dimension(errors)

    return d_estimate, errors, embedding


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

    # Get upper triangle indices (exclude diagonal)
    n = D_original.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    D_orig_flat = D_original[triu_idx]

    # Normalize by max distance for scale-invariance
    D_max = np.max(D_orig_flat)
    if D_max == 0:
        return np.zeros(k_max)

    for k in range(1, k_max + 1):
        # Reconstruct distances from k-dimensional embedding
        emb_k = embedding[:, :k]
        D_recon = squareform(pdist(emb_k))
        D_recon_flat = D_recon[triu_idx]

        # Normalize reconstructed distances
        recon_max = np.max(D_recon_flat)
        if recon_max > 0:
            D_recon_flat = D_recon_flat * (D_max / recon_max)

        # Compute normalized RMSE
        errors[k - 1] = np.sqrt(np.mean((D_orig_flat - D_recon_flat) ** 2)) / D_max

    return errors


def find_intrinsic_dimension(errors: np.ndarray, threshold: float = 0.1) -> int:
    """
    Find intrinsic dimension from error curve.

    Dimension is the first k where error drops below threshold * initial_error.

    Args:
        errors: Array of reconstruction errors for each dimension
        threshold: Fraction of initial error to use as cutoff

    Returns:
        Estimated intrinsic dimension
    """
    if len(errors) == 0:
        return 1

    # Get threshold relative to first dimension's error
    initial_error = errors[0]
    if initial_error == 0:
        return 1

    cutoff = threshold * initial_error

    # Find first dimension below threshold
    below_threshold = np.where(errors < cutoff)[0]

    if len(below_threshold) > 0:
        return below_threshold[0] + 1  # +1 because dimensions are 1-indexed
    else:
        # If never below threshold, find elbow using second derivative
        return find_elbow(errors)


def find_elbow(errors: np.ndarray) -> int:
    """
    Find elbow point in error curve using curvature.

    Args:
        errors: Array of reconstruction errors

    Returns:
        Dimension at elbow point
    """
    if len(errors) < 3:
        return len(errors)

    # Compute second derivative (discrete)
    d2 = np.diff(np.diff(errors))

    # Elbow is where second derivative is maximized (most negative to less negative)
    if len(d2) == 0:
        return 1

    elbow_idx = np.argmax(d2) + 1  # +1 because diff reduces length

    return max(1, min(elbow_idx + 1, len(errors)))  # +1 for 1-indexing


def validate_methods(d_isomap: int, d_mds: int, max_diff: float = 0.5) -> bool:
    """
    Check if Isomap and MDS dimension estimates agree.

    Args:
        d_isomap: Dimension from Isomap
        d_mds: Dimension from MDS
        max_diff: Maximum allowed difference

    Returns:
        True if methods agree within tolerance
    """
    return abs(d_isomap - d_mds) <= max_diff


def extract_dimension_both(D: np.ndarray, k_max: int = 10,
                           n_neighbors: int = 8) -> dict:
    """
    Extract dimension using both Isomap and MDS, with validation.

    Args:
        D: Distance matrix
        k_max: Maximum dimension to test
        n_neighbors: Number of neighbors for Isomap

    Returns:
        Dictionary with:
        - d_isomap: Isomap dimension estimate
        - d_mds: MDS dimension estimate
        - d_mean: Average of both
        - valid: Whether methods agree
        - errors_isomap: Isomap error curve
        - errors_mds: MDS error curve
        - embedding_isomap: Isomap embedding
        - embedding_mds: MDS embedding
    """
    d_isomap, errors_isomap, emb_isomap = extract_dimension_isomap(D, k_max, n_neighbors)
    d_mds, errors_mds, emb_mds = extract_dimension_mds(D, k_max)

    valid = validate_methods(d_isomap, d_mds)
    d_mean = (d_isomap + d_mds) / 2

    return {
        'd_isomap': d_isomap,
        'd_mds': d_mds,
        'd_mean': d_mean,
        'valid': valid,
        'errors_isomap': errors_isomap,
        'errors_mds': errors_mds,
        'embedding_isomap': emb_isomap,
        'embedding_mds': emb_mds
    }


def normalize_distances(D: np.ndarray) -> np.ndarray:
    """
    Normalize distance matrix to [0, 1] range.

    Args:
        D: Distance matrix

    Returns:
        Normalized distance matrix
    """
    D_max = np.max(D[np.isfinite(D)])
    if D_max == 0:
        return D
    return D / D_max
