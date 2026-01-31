"""
Quantum mutual information computation for Experiment 04.

Provides functions to compute:
- Partial traces (reduced density matrices)
- Von Neumann entropy
- Quantum mutual information between qubit pairs
- MI-based distance matrices
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings


def partial_trace(rho: np.ndarray, keep_qubits: List[int], n_qubits: int) -> np.ndarray:
    """
    Compute partial trace, keeping only specified qubits.

    Args:
        rho: Density matrix (2^n x 2^n)
        keep_qubits: List of qubit indices to keep (0-indexed)
        n_qubits: Total number of qubits

    Returns:
        Reduced density matrix
    """
    if len(keep_qubits) == n_qubits:
        return rho.copy()

    if len(keep_qubits) == 0:
        return np.array([[np.trace(rho)]])

    dim = 2 ** n_qubits
    keep_dim = 2 ** len(keep_qubits)
    trace_qubits = [q for q in range(n_qubits) if q not in keep_qubits]

    # Reshape to tensor form
    rho_tensor = rho.reshape([2] * (2 * n_qubits))

    # Trace out unwanted qubits
    # We need to contract pairs of indices for each traced qubit
    for q in sorted(trace_qubits, reverse=True):
        # In tensor notation, qubit q has indices q (ket) and q + n_qubits (bra)
        # After each trace, indices shift, so we need to track carefully
        current_n = rho_tensor.ndim // 2
        ket_idx = q
        bra_idx = q + current_n

        # Trace: contract ket_idx with bra_idx
        rho_tensor = np.trace(rho_tensor, axis1=ket_idx, axis2=bra_idx)

    # Reshape back to matrix
    reduced_rho = rho_tensor.reshape(keep_dim, keep_dim)

    return reduced_rho


def partial_trace_fast(rho: np.ndarray, keep_qubits: List[int], n_qubits: int) -> np.ndarray:
    """
    Fast partial trace implementation using einsum for small systems.

    More efficient for small n_qubits.
    """
    if len(keep_qubits) == n_qubits:
        return rho.copy()

    if len(keep_qubits) == 0:
        return np.array([[np.trace(rho)]])

    dim = 2 ** n_qubits
    keep_dim = 2 ** len(keep_qubits)
    trace_qubits = sorted([q for q in range(n_qubits) if q not in keep_qubits])

    # For efficiency with small systems, use direct computation
    reduced = np.zeros((keep_dim, keep_dim), dtype=np.complex128)

    for i in range(keep_dim):
        for j in range(keep_dim):
            # Convert i, j to bit strings for kept qubits
            i_bits = [(i >> (len(keep_qubits) - 1 - k)) & 1 for k in range(len(keep_qubits))]
            j_bits = [(j >> (len(keep_qubits) - 1 - k)) & 1 for k in range(len(keep_qubits))]

            # Sum over traced qubits
            for trace_val in range(2 ** len(trace_qubits)):
                trace_bits = [(trace_val >> (len(trace_qubits) - 1 - k)) & 1
                              for k in range(len(trace_qubits))]

                # Build full index for ket (row) and bra (col)
                row_bits = [0] * n_qubits
                col_bits = [0] * n_qubits

                # Fill in kept qubit values
                for k, q in enumerate(sorted(keep_qubits)):
                    row_bits[q] = i_bits[k]
                    col_bits[q] = j_bits[k]

                # Fill in traced qubit values (same for row and col)
                for k, q in enumerate(trace_qubits):
                    row_bits[q] = trace_bits[k]
                    col_bits[q] = trace_bits[k]

                # Convert to indices
                row_idx = sum(b << (n_qubits - 1 - q) for q, b in enumerate(row_bits))
                col_idx = sum(b << (n_qubits - 1 - q) for q, b in enumerate(col_bits))

                reduced[i, j] += rho[row_idx, col_idx]

    return reduced


def von_neumann_entropy(rho: np.ndarray, base: int = 2) -> float:
    """
    Compute von Neumann entropy S(rho) = -Tr(rho log rho).

    Args:
        rho: Density matrix
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy value
    """
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho)

    # Filter out small/negative eigenvalues (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    if len(eigenvalues) == 0:
        return 0.0

    # Compute entropy
    if base == 2:
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    else:
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    return float(entropy)


def mutual_information(rho: np.ndarray, qubit_i: int, qubit_j: int,
                        n_qubits: int) -> float:
    """
    Compute quantum mutual information I(i:j) between two qubits.

    I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB)

    Args:
        rho: Full density matrix
        qubit_i: First qubit index
        qubit_j: Second qubit index
        n_qubits: Total number of qubits

    Returns:
        Mutual information in bits
    """
    # Get reduced density matrices
    rho_i = partial_trace_fast(rho, [qubit_i], n_qubits)
    rho_j = partial_trace_fast(rho, [qubit_j], n_qubits)
    rho_ij = partial_trace_fast(rho, sorted([qubit_i, qubit_j]), n_qubits)

    # Compute entropies
    S_i = von_neumann_entropy(rho_i)
    S_j = von_neumann_entropy(rho_j)
    S_ij = von_neumann_entropy(rho_ij)

    # Mutual information
    mi = S_i + S_j - S_ij

    return mi


def mutual_information_matrix(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Compute mutual information matrix for all qubit pairs.

    Args:
        rho: Density matrix
        n_qubits: Number of qubits

    Returns:
        N x N symmetric matrix of mutual information values
    """
    MI = np.zeros((n_qubits, n_qubits))

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            mi = mutual_information(rho, i, j, n_qubits)
            MI[i, j] = mi
            MI[j, i] = mi

    return MI


def mi_to_distance(MI: np.ndarray, method: str = 'sqrt') -> np.ndarray:
    """
    Convert mutual information matrix to distance matrix.

    Args:
        MI: Mutual information matrix
        method: Conversion method
            - 'sqrt': D = sqrt(2 * (S_max - MI)) where S_max = 1 for qubits
            - 'inverse': D = 1 / (MI + epsilon)
            - 'negative': D = S_max - MI

    Returns:
        Distance matrix
    """
    n = MI.shape[0]
    S_max = 1.0  # Max entropy for single qubit (log2(2) = 1)

    if method == 'sqrt':
        # D = sqrt(2 * (S_max - MI))
        # Maps MI=S_max -> D=0, MI=0 -> D=sqrt(2)
        D = np.sqrt(2 * np.maximum(0, S_max - MI))

    elif method == 'inverse':
        # D = 1 / (MI + epsilon)
        epsilon = 0.01
        D = 1.0 / (MI + epsilon)

    elif method == 'negative':
        # D = S_max - MI
        D = S_max - MI

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure diagonal is 0
    np.fill_diagonal(D, 0)

    return D


def entanglement_entropy(rho: np.ndarray, partition: List[int], n_qubits: int) -> float:
    """
    Compute bipartite entanglement entropy.

    S_ent = S(rho_A) where A is the partition subsystem.

    Args:
        rho: Full density matrix
        partition: List of qubits in subsystem A
        n_qubits: Total number of qubits

    Returns:
        Entanglement entropy
    """
    rho_A = partial_trace_fast(rho, partition, n_qubits)
    return von_neumann_entropy(rho_A)


def half_chain_entanglement(rho: np.ndarray, n_qubits: int) -> float:
    """
    Compute entanglement entropy for half-chain bipartition.

    Standard measure of entanglement for 1D systems.

    Args:
        rho: Density matrix
        n_qubits: Number of qubits

    Returns:
        Half-chain entanglement entropy
    """
    half = n_qubits // 2
    partition = list(range(half))
    return entanglement_entropy(rho, partition, n_qubits)


def validate_mi_matrix(MI: np.ndarray) -> dict:
    """
    Validate mutual information matrix properties.

    Returns:
        Validation results dictionary
    """
    n = MI.shape[0]

    # Check symmetry
    is_symmetric = np.allclose(MI, MI.T)

    # Check non-negativity
    min_val = np.min(MI)
    is_nonnegative = min_val >= -1e-10

    # Check diagonal (should be 0 for MI between same qubit - not applicable)
    diagonal_zero = np.allclose(np.diag(MI), 0)

    # Max value (should be <= 2 for qubits)
    max_val = np.max(MI)
    max_reasonable = max_val <= 2.1  # Allow small numerical error

    return {
        'is_symmetric': is_symmetric,
        'is_nonnegative': is_nonnegative,
        'diagonal_zero': diagonal_zero,
        'max_reasonable': max_reasonable,
        'min_value': float(min_val),
        'max_value': float(max_val),
        'mean_value': float(np.mean(MI[np.triu_indices(n, k=1)]))
    }
