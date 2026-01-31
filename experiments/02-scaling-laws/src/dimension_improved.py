"""
Improved dimension estimation module for Experiment 02.

Provides continuous dimension estimation using error curves,
and sparse methods for large graphs (N > 500).

Key improvement over Exp01: Returns fractional dimensions based on
interpolation of error curves, rather than integer threshold-based detection.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.manifold import Isomap, MDS
from typing import Tuple, Optional, List
import warnings


def compute_reconstruction_errors(D_original: np.ndarray, embedding: np.ndarray,
                                   k_max: int) -> np.ndarray:
    """
    Compute reconstruction error for each embedding dimension.
    Identical to Exp01 for consistency.

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
        if k > embedding.shape[1]:
            errors[k - 1] = errors[k - 2] if k > 1 else 1.0
            continue

        emb_k = embedding[:, :k]
        D_recon = squareform(pdist(emb_k))
        D_recon_flat = D_recon[triu_idx]

        recon_max = np.max(D_recon_flat)
        if recon_max > 0:
            D_recon_flat = D_recon_flat * (D_max / recon_max)

        errors[k - 1] = np.sqrt(np.mean((D_orig_flat - D_recon_flat) ** 2)) / D_max

    return errors


def explained_variance_curve(D_original: np.ndarray, embedding: np.ndarray,
                             k_max: int = 10) -> np.ndarray:
    """
    Compute explained variance for each embedding dimension.
    Converts reconstruction errors to explained variance.

    Args:
        D_original: Original N x N distance matrix
        embedding: N x k_max embedding
        k_max: Maximum dimension to compute

    Returns:
        Array of explained variance ratios for dimensions 1..k_max
    """
    errors = compute_reconstruction_errors(D_original, embedding, k_max)
    # Convert errors to explained variance: 1 - error
    variances = np.maximum(0.0, 1.0 - errors)
    return variances


def continuous_dimension(D: np.ndarray, k_max: int = 10,
                         n_neighbors: int = 8,
                         error_threshold: float = 0.1) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute continuous (fractional) dimension based on error curve.

    Uses same approach as Exp01 but with fractional interpolation.

    Args:
        D: N x N distance matrix (precomputed)
        k_max: Maximum dimension to test
        n_neighbors: Number of neighbors for Isomap
        error_threshold: Fraction of initial error to use as cutoff (same as Exp01)

    Returns:
        Tuple of (fractional dimension, error curve, embedding)
    """
    n = D.shape[0]
    n_neighbors = min(n_neighbors, n - 1)
    k_max = min(k_max, n - 1)

    # Fit Isomap with maximum dimensions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isomap = Isomap(n_components=k_max, n_neighbors=n_neighbors,
                        metric='precomputed')
        try:
            embedding = isomap.fit_transform(D)
        except Exception:
            return float(k_max), np.ones(k_max), np.zeros((n, k_max))

    # Compute reconstruction errors (same as Exp01)
    errors = compute_reconstruction_errors(D, embedding, k_max)

    # Find fractional dimension using interpolation on error curve
    d_frac = find_fractional_dimension_from_errors(errors, error_threshold)

    return d_frac, errors, embedding


def find_fractional_dimension_from_errors(errors: np.ndarray,
                                          threshold: float = 0.1) -> float:
    """
    Find fractional dimension from error curve using interpolation.

    Dimension is where error drops below threshold * initial_error.
    Uses linear interpolation between integer dimensions.

    Args:
        errors: Array of reconstruction errors for each dimension
        threshold: Fraction of initial error to use as cutoff

    Returns:
        Fractional dimension estimate
    """
    if len(errors) == 0 or errors[0] == 0:
        return 1.0

    initial_error = errors[0]
    cutoff = threshold * initial_error

    # Find first dimension below threshold
    below_threshold = np.where(errors < cutoff)[0]

    if len(below_threshold) > 0:
        first_below = below_threshold[0]
        if first_below == 0:
            return 1.0

        # Linear interpolation between (first_below-1) and first_below
        e_above = errors[first_below - 1]
        e_below = errors[first_below]

        if e_above == e_below:
            return float(first_below + 1)

        # Interpolate: find where we cross cutoff
        frac = (e_above - cutoff) / (e_above - e_below)
        d_frac = first_below + frac  # dimensions are 1-indexed

        return d_frac
    else:
        # Never below threshold - use elbow detection with interpolation
        return find_elbow_fractional(errors)


def find_fractional_dimension(variances: np.ndarray, threshold: float = 0.9) -> float:
    """
    Find fractional dimension where threshold variance is explained.

    Uses linear interpolation between integer dimensions.

    Args:
        variances: Explained variance at each dimension
        threshold: Target variance threshold

    Returns:
        Fractional dimension estimate
    """
    if len(variances) == 0:
        return 1.0

    # Find first dimension exceeding threshold
    above_threshold = np.where(variances >= threshold)[0]

    if len(above_threshold) == 0:
        # Never reaches threshold - return dimension with max variance
        return float(len(variances))

    first_above = above_threshold[0]

    if first_above == 0:
        # Already above threshold at d=1
        return 1.0

    # Linear interpolation between (first_above-1) and first_above
    v_below = variances[first_above - 1]
    v_above = variances[first_above]

    if v_above == v_below:
        return float(first_above + 1)

    # Interpolate: find where we cross threshold
    frac = (threshold - v_below) / (v_above - v_below)
    d_frac = first_above + frac  # +1 for 1-indexing, -1 because first_above is 0-indexed

    return d_frac


def find_elbow_fractional(errors: np.ndarray) -> float:
    """
    Find elbow point in error curve with fractional precision.

    Args:
        errors: Array of reconstruction errors

    Returns:
        Fractional dimension at elbow point
    """
    if len(errors) < 3:
        return float(len(errors))

    # Compute second derivative (discrete)
    d2 = np.diff(np.diff(errors))

    if len(d2) == 0:
        return 1.0

    # Elbow is where second derivative is maximized
    elbow_idx = np.argmax(d2)

    # Use parabolic interpolation for fractional estimate
    if 0 < elbow_idx < len(d2) - 1:
        # Fit parabola to 3 points around maximum
        y0, y1, y2 = d2[elbow_idx - 1], d2[elbow_idx], d2[elbow_idx + 1]
        # Vertex of parabola through these points
        denom = 2 * (y0 - 2 * y1 + y2)
        if denom != 0:
            offset = (y0 - y2) / denom
            offset = max(-0.5, min(0.5, offset))  # Clamp to reasonable range
        else:
            offset = 0
        d_frac = elbow_idx + 1 + offset + 1  # +2 because diff reduces length twice
    else:
        d_frac = elbow_idx + 2  # +2 for 1-indexing and diff reduction

    return max(1.0, min(float(len(errors)), d_frac))


def landmark_mds(D: np.ndarray, n_landmarks: int = 200,
                 n_components: int = 10,
                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Landmark MDS for large distance matrices.

    Selects landmarks, embeds them with classical MDS, then projects
    remaining points using distance-based triangulation.

    Args:
        D: N x N distance matrix
        n_landmarks: Number of landmark points
        n_components: Embedding dimension
        seed: Random seed for landmark selection

    Returns:
        Tuple of (embedding, landmark_indices)
    """
    n = D.shape[0]
    n_landmarks = min(n_landmarks, n)
    n_components = min(n_components, n_landmarks - 1)

    np.random.seed(seed)

    # Select landmarks (uniform random)
    landmark_idx = np.random.choice(n, size=n_landmarks, replace=False)
    landmark_idx = np.sort(landmark_idx)

    # Extract landmark distance matrix
    D_landmarks = D[np.ix_(landmark_idx, landmark_idx)]

    # Classical MDS on landmarks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mds = MDS(n_components=n_components, dissimilarity='precomputed',
                  normalized_stress='auto', random_state=seed, max_iter=500)
        try:
            L = mds.fit_transform(D_landmarks)
        except Exception:
            return np.zeros((n, n_components)), landmark_idx

    # Project other points using triangulation
    embedding = np.zeros((n, n_components))
    embedding[landmark_idx] = L

    # For non-landmark points, use weighted average based on distances to landmarks
    non_landmark_mask = np.ones(n, dtype=bool)
    non_landmark_mask[landmark_idx] = False

    if np.any(non_landmark_mask):
        D_to_landmarks = D[non_landmark_mask][:, landmark_idx]

        # Use inverse distance weighting (with epsilon for stability)
        eps = 1e-10
        weights = 1.0 / (D_to_landmarks + eps)
        weights = weights / weights.sum(axis=1, keepdims=True)

        embedding[non_landmark_mask] = weights @ L

    return embedding, landmark_idx


def sparse_isomap(D: np.ndarray, k: int = 15, n_components: int = 10,
                  error_threshold: float = 0.1) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Sparse Isomap using k-NN graph instead of full distance matrix.

    For large N, only uses k nearest neighbors to build the geodesic
    distance approximation.

    Args:
        D: N x N distance matrix
        k: Number of nearest neighbors
        n_components: Maximum embedding dimension
        error_threshold: Threshold for dimension estimation

    Returns:
        Tuple of (fractional dimension, error curve, embedding)
    """
    n = D.shape[0]
    k = min(k, n - 1)
    n_components = min(n_components, n - 1)

    # Build sparse k-NN graph
    D_sparse = np.full_like(D, np.inf)
    np.fill_diagonal(D_sparse, 0)

    for i in range(n):
        # Find k nearest neighbors
        neighbors = np.argsort(D[i])[:k + 1]  # +1 to include self
        for j in neighbors:
            if i != j:
                D_sparse[i, j] = D[i, j]
                D_sparse[j, i] = D[j, i]

    # Use sklearn Isomap with sparse neighbor graph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isomap = Isomap(n_components=n_components, n_neighbors=k,
                        metric='precomputed')
        try:
            embedding = isomap.fit_transform(D)
        except Exception:
            return float(n_components), np.ones(n_components), np.zeros((n, n_components))

    # Compute reconstruction errors
    errors = compute_reconstruction_errors(D, embedding, n_components)
    d_frac = find_fractional_dimension_from_errors(errors, error_threshold)

    return d_frac, errors, embedding


def continuous_dimension_both(D: np.ndarray, k_max: int = 10,
                              n_neighbors: int = 8,
                              use_sparse: bool = False,
                              error_threshold: float = 0.1) -> dict:
    """
    Compute continuous dimension using both Isomap and MDS.

    Args:
        D: Distance matrix
        k_max: Maximum dimension to test
        n_neighbors: Number of neighbors for Isomap
        use_sparse: Whether to use sparse methods (for large N)
        error_threshold: Threshold for dimension estimation (same as Exp01)

    Returns:
        Dictionary with dimension estimates and error/variance curves
    """
    n = D.shape[0]
    k_max = min(k_max, n - 1)

    # Isomap
    if use_sparse and n > 500:
        d_isomap, errors_isomap, emb_isomap = sparse_isomap(D, k=n_neighbors, n_components=k_max)
    else:
        d_isomap, errors_isomap, emb_isomap = continuous_dimension(D, k_max, n_neighbors, error_threshold)

    # MDS (use landmark for large N)
    if use_sparse and n > 500:
        emb_mds, _ = landmark_mds(D, n_landmarks=min(200, n), n_components=k_max)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mds = MDS(n_components=k_max, dissimilarity='precomputed',
                      normalized_stress='auto', random_state=42, max_iter=300)
            try:
                emb_mds = mds.fit_transform(D)
            except Exception:
                emb_mds = np.zeros((n, k_max))

    # Compute errors for MDS
    errors_mds = compute_reconstruction_errors(D, emb_mds, k_max)
    d_mds = find_fractional_dimension_from_errors(errors_mds, error_threshold)

    d_mean = (d_isomap + d_mds) / 2

    # Convert errors to variances for compatibility
    variances_isomap = np.maximum(0.0, 1.0 - errors_isomap)
    variances_mds = np.maximum(0.0, 1.0 - errors_mds)

    return {
        'd_isomap': d_isomap,
        'd_mds': d_mds,
        'd_mean': d_mean,
        'variances_isomap': variances_isomap,
        'variances_mds': variances_mds,
        'embedding_isomap': emb_isomap,
        'embedding_mds': emb_mds
    }
