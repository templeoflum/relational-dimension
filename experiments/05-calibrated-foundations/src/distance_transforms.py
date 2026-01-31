"""
Distance transformation functions.

Converts correlation matrices to distance matrices using various methods.
The calibration phase will select the best transformation.
"""

import numpy as np
from typing import Callable, Dict, List


def transform_sqrt(C: np.ndarray) -> np.ndarray:
    """
    Standard sqrt transformation: D = sqrt(2 * (1 - C))

    Maps C=1 -> D=0, C=0 -> D=sqrt(2), C=-1 -> D=2
    """
    # Clip to valid range
    C_clipped = np.clip(C, -1, 1)
    D = np.sqrt(2 * (1 - C_clipped))
    np.fill_diagonal(D, 0)
    return D


def transform_linear(C: np.ndarray) -> np.ndarray:
    """
    Linear transformation: D = 1 - C

    Maps C=1 -> D=0, C=0 -> D=1, C=-1 -> D=2
    """
    C_clipped = np.clip(C, -1, 1)
    D = 1 - C_clipped
    np.fill_diagonal(D, 0)
    return D


def transform_arccos(C: np.ndarray) -> np.ndarray:
    """
    Arccos transformation: D = arccos(C)

    Maps C=1 -> D=0, C=0 -> D=pi/2, C=-1 -> D=pi
    This is the angular distance in a unit sphere.
    """
    C_clipped = np.clip(C, -1, 1)
    D = np.arccos(C_clipped)
    np.fill_diagonal(D, 0)
    return D


def transform_neglog(C: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """
    Negative log transformation: D = -log(C + epsilon)

    Maps high C -> low D (logarithmically)
    """
    C_shifted = np.clip(C + epsilon, epsilon, 1 + epsilon)
    D = -np.log(C_shifted)
    # Normalize to [0, max]
    D = D - D.min()
    np.fill_diagonal(D, 0)
    return D


def transform_sqrt_normalized(C: np.ndarray) -> np.ndarray:
    """
    Normalized sqrt: D = sqrt(1 - C) (without the factor of 2)

    Maps C=1 -> D=0, C=0 -> D=1
    """
    C_clipped = np.clip(C, -1, 1)
    D = np.sqrt(1 - C_clipped)
    np.fill_diagonal(D, 0)
    return D


# Registry of all transforms
TRANSFORMS: Dict[str, Callable] = {
    'sqrt': transform_sqrt,
    'linear': transform_linear,
    'arccos': transform_arccos,
    'neglog': transform_neglog,
    'sqrt_norm': transform_sqrt_normalized,
}


def get_transform(name: str) -> Callable:
    """Get a transform function by name."""
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORMS.keys())}")
    return TRANSFORMS[name]


def list_transforms() -> List[str]:
    """List available transform names."""
    return list(TRANSFORMS.keys())


def apply_transform(C: np.ndarray, name: str) -> np.ndarray:
    """Apply a named transform to correlation matrix."""
    transform_fn = get_transform(name)
    return transform_fn(C)


def validate_distance_matrix(D: np.ndarray, tol: float = 1e-10) -> Dict[str, bool]:
    """
    Validate properties of a distance matrix.

    Returns dict with validation results.
    """
    n = D.shape[0]

    # Check symmetry
    is_symmetric = np.allclose(D, D.T, atol=tol)

    # Check non-negativity
    is_nonnegative = np.all(D >= -tol)

    # Check diagonal is zero
    diagonal_zero = np.allclose(np.diag(D), 0, atol=tol)

    # Check triangle inequality (sample-based for efficiency)
    triangle_violations = 0
    n_checks = min(1000, n * (n-1) * (n-2) // 6)

    if n >= 3:
        for _ in range(n_checks):
            i, j, k = np.random.choice(n, 3, replace=False)
            if D[i, j] > D[i, k] + D[k, j] + tol:
                triangle_violations += 1

    triangle_ok = triangle_violations < n_checks * 0.01  # <1% violations

    return {
        'is_symmetric': is_symmetric,
        'is_nonnegative': is_nonnegative,
        'diagonal_zero': diagonal_zero,
        'triangle_inequality': triangle_ok,
        'triangle_violations': triangle_violations,
        'is_valid_metric': is_symmetric and is_nonnegative and diagonal_zero and triangle_ok
    }
