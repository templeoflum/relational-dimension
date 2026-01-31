"""
Lattice generation for calibration testing.

Generates regular lattice structures with known intrinsic dimensions:
- 1D chain (d=1)
- 2D square lattice (d=2)
- 3D cubic lattice (d=3)
"""

import numpy as np
from typing import Tuple, Dict, Any
from scipy.spatial.distance import pdist, squareform


def generate_1d_chain(n_nodes: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate 1D chain lattice.

    Args:
        n_nodes: Number of nodes

    Returns:
        positions: (n_nodes, 1) array of positions
        distances: (n_nodes, n_nodes) distance matrix
        metadata: Dictionary with lattice properties
    """
    positions = np.arange(n_nodes).reshape(-1, 1).astype(float)
    distances = squareform(pdist(positions))

    metadata = {
        'type': '1d_chain',
        'n_nodes': n_nodes,
        'true_dimension': 1,
        'shape': (n_nodes,)
    }

    return positions, distances, metadata


def generate_2d_lattice(side: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate 2D square lattice.

    Args:
        side: Side length (total nodes = side^2)

    Returns:
        positions: (n_nodes, 2) array of positions
        distances: (n_nodes, n_nodes) distance matrix
        metadata: Dictionary with lattice properties
    """
    n_nodes = side * side
    positions = np.zeros((n_nodes, 2))

    for i in range(side):
        for j in range(side):
            idx = i * side + j
            positions[idx] = [i, j]

    distances = squareform(pdist(positions))

    metadata = {
        'type': '2d_lattice',
        'n_nodes': n_nodes,
        'true_dimension': 2,
        'shape': (side, side)
    }

    return positions, distances, metadata


def generate_3d_lattice(dims: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate 3D cubic lattice.

    Args:
        dims: Tuple of (x, y, z) dimensions

    Returns:
        positions: (n_nodes, 3) array of positions
        distances: (n_nodes, n_nodes) distance matrix
        metadata: Dictionary with lattice properties
    """
    nx, ny, nz = dims
    n_nodes = nx * ny * nz
    positions = np.zeros((n_nodes, 3))

    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                positions[idx] = [i, j, k]
                idx += 1

    distances = squareform(pdist(positions))

    metadata = {
        'type': '3d_lattice',
        'n_nodes': n_nodes,
        'true_dimension': 3,
        'shape': dims
    }

    return positions, distances, metadata


def distances_to_correlation(D: np.ndarray, method: str = 'gaussian',
                             scale: float = None) -> np.ndarray:
    """
    Convert distance matrix to correlation matrix.

    This creates a correlation matrix that matches the topology,
    used for calibration (should give delta ≈ 0).

    Args:
        D: Distance matrix
        method: 'gaussian' or 'inverse'
        scale: Scale parameter (default: median distance)

    Returns:
        Correlation matrix
    """
    if scale is None:
        # Use median non-zero distance as scale
        D_flat = D[np.triu_indices_from(D, k=1)]
        scale = np.median(D_flat[D_flat > 0])

    if method == 'gaussian':
        # C = exp(-D^2 / (2 * scale^2))
        C = np.exp(-D**2 / (2 * scale**2))
    elif method == 'inverse':
        # C = 1 / (1 + D/scale)
        C = 1 / (1 + D / scale)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure diagonal is 1
    np.fill_diagonal(C, 1.0)

    # Ensure symmetry
    C = (C + C.T) / 2

    return C


def generate_calibration_system(system_type: str, n_nodes: int = 100) -> Dict[str, Any]:
    """
    Generate a complete calibration system with matched topology and correlation.

    Args:
        system_type: '1d', '2d', or '3d'
        n_nodes: Approximate number of nodes

    Returns:
        Dictionary with positions, distances, correlation, and metadata
    """
    if system_type == '1d':
        positions, D_topo, metadata = generate_1d_chain(n_nodes)
    elif system_type == '2d':
        side = int(np.sqrt(n_nodes))
        positions, D_topo, metadata = generate_2d_lattice(side)
    elif system_type == '3d':
        # Approximate cube root
        side = int(np.cbrt(n_nodes))
        dims = (side, side, max(1, n_nodes // (side * side)))
        positions, D_topo, metadata = generate_3d_lattice(dims)
    else:
        raise ValueError(f"Unknown system type: {system_type}")

    # Create matched correlation matrix
    C = distances_to_correlation(D_topo, method='gaussian')

    return {
        'positions': positions,
        'D_topo': D_topo,
        'C': C,
        'metadata': metadata
    }
