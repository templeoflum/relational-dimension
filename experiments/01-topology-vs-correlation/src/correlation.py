"""
Correlation matrix generation module for Experiment 01.

Provides functions to create different correlation patterns:
- Nearest-neighbor (NN): correlations only between adjacent nodes
- Long-range (LR): exponential decay based on Euclidean distance
- Random (RAND): random pair correlations for control
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Optional


def correlation_nn(adjacency: np.ndarray, rho: float = 0.8) -> np.ndarray:
    """
    Create nearest-neighbor only correlations.

    Correlation is rho for adjacent nodes, 0 otherwise.

    Args:
        adjacency: N x N adjacency matrix
        rho: Correlation strength for neighbors (0 < rho < 1)

    Returns:
        N x N correlation matrix (positive semidefinite)
    """
    n = adjacency.shape[0]

    # Start with identity (self-correlation = 1)
    C = np.eye(n)

    # Add correlations for neighbors
    C += rho * adjacency

    # Ensure positive semidefiniteness
    C = ensure_positive_semidefinite(C)

    return C


def correlation_lr(positions: np.ndarray, rho: float = 0.8,
                   lambda_corr: Optional[float] = None) -> np.ndarray:
    """
    Create long-range correlations with exponential decay.

    C[i,j] = rho * exp(-d_euclidean(i,j) / lambda)

    Args:
        positions: N x 2 array of node positions
        rho: Maximum correlation strength
        lambda_corr: Correlation length scale (default: 0.3 * diameter)

    Returns:
        N x N correlation matrix (positive semidefinite)
    """
    from scipy.spatial.distance import cdist

    n = positions.shape[0]

    # Compute Euclidean distances
    distances = cdist(positions, positions)

    # Default lambda is 0.3 times the diameter
    if lambda_corr is None:
        diameter = np.max(distances)
        lambda_corr = 0.3 * diameter

    # Exponential decay correlation
    C = rho * np.exp(-distances / lambda_corr)

    # Set diagonal to 1 (self-correlation)
    np.fill_diagonal(C, 1.0)

    # Ensure positive semidefiniteness
    C = ensure_positive_semidefinite(C)

    return C


def correlation_rand(n: int, k: int, rho: float = 0.8,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Create random pair correlations.

    Selects k random pairs and sets their correlation to rho.

    Args:
        n: Number of nodes
        k: Number of random pairs to correlate
        rho: Correlation strength
        seed: Random seed for reproducibility

    Returns:
        N x N correlation matrix (positive semidefinite)
    """
    if seed is not None:
        np.random.seed(seed)

    # Start with identity
    C = np.eye(n)

    # Generate random pairs (upper triangle indices)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Limit k to available pairs
    k = min(k, len(all_pairs))

    # Select random pairs
    selected_idx = np.random.choice(len(all_pairs), size=k, replace=False)
    selected_pairs = [all_pairs[i] for i in selected_idx]

    # Set correlations
    for i, j in selected_pairs:
        C[i, j] = rho
        C[j, i] = rho

    # Ensure positive semidefiniteness
    C = ensure_positive_semidefinite(C)

    return C


def ensure_positive_semidefinite(C: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Project matrix to nearest positive semidefinite matrix.

    Uses eigenvalue decomposition and clips negative eigenvalues.

    Args:
        C: Input matrix (should be symmetric)
        epsilon: Minimum eigenvalue threshold

    Returns:
        Positive semidefinite matrix
    """
    # Ensure symmetry
    C = (C + C.T) / 2

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Reconstruct matrix
    C_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Ensure diagonal is exactly 1 (for correlation matrix)
    d = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / np.outer(d, d)

    # Force exact symmetry and unit diagonal
    C_psd = (C_psd + C_psd.T) / 2
    np.fill_diagonal(C_psd, 1.0)

    return C_psd


def correlation_to_distance(C: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.

    D[i,j] = sqrt(2 * (1 - C[i,j]))

    This maps:
    - C = 1 (perfect correlation) -> D = 0
    - C = 0 (no correlation) -> D = sqrt(2) ≈ 1.41
    - C = -1 (anticorrelation) -> D = 2

    Args:
        C: Correlation matrix

    Returns:
        Distance matrix
    """
    # Ensure correlations are in valid range
    C_clipped = np.clip(C, -1, 1)

    # Convert to distance
    D = np.sqrt(2 * (1 - C_clipped))

    # Ensure diagonal is 0
    np.fill_diagonal(D, 0)

    return D


def count_correlations(C: np.ndarray, threshold: float = 0.1) -> int:
    """
    Count number of significant correlations in matrix.

    Args:
        C: Correlation matrix
        threshold: Minimum correlation to count

    Returns:
        Number of pairs with |C[i,j]| > threshold
    """
    n = C.shape[0]
    # Upper triangle only (exclude diagonal)
    upper = np.triu(np.abs(C), k=1)
    return np.sum(upper > threshold)


def estimate_k_for_density(positions: np.ndarray, rho: float,
                           lambda_corr: Optional[float] = None,
                           threshold: float = 0.1) -> int:
    """
    Estimate number of random pairs needed to match LR correlation density.

    Args:
        positions: Node positions for LR correlation reference
        rho: Correlation strength
        lambda_corr: Correlation length scale
        threshold: Minimum correlation to count

    Returns:
        Number of pairs k for random correlation
    """
    # Generate LR correlation matrix
    C_lr = correlation_lr(positions, rho, lambda_corr)

    # Count significant correlations
    return count_correlations(C_lr, threshold)
