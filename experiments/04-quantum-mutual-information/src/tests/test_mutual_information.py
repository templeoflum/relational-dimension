"""Tests for quantum mutual information computation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_states import (
    product_state, ghz_state, w_state, state_to_density_matrix
)
from mutual_information import (
    partial_trace, partial_trace_fast, von_neumann_entropy,
    mutual_information, mutual_information_matrix,
    mi_to_distance, validate_mi_matrix,
    entanglement_entropy, half_chain_entanglement
)


class TestPartialTrace:
    """Tests for partial trace operations."""

    def test_full_trace(self):
        """Trace over all qubits should give scalar trace."""
        state = ghz_state(3)
        rho = state_to_density_matrix(state)
        result = partial_trace(rho, [], 3)
        # Should be 1x1 matrix with trace = 1
        assert result.shape == (1, 1)
        assert np.abs(result[0, 0] - 1.0) < 1e-10

    def test_no_trace(self):
        """Trace over no qubits should return original."""
        state = ghz_state(3)
        rho = state_to_density_matrix(state)
        result = partial_trace(rho, [0, 1, 2], 3)
        assert np.allclose(rho, result)

    def test_single_qubit_trace(self):
        """Trace to single qubit should give 2x2 matrix."""
        state = product_state(3)
        rho = state_to_density_matrix(state)
        result = partial_trace_fast(rho, [0], 3)
        # For |000>, tracing out qubits 1,2 should give |0><0|
        assert result.shape == (2, 2)
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert np.allclose(result, expected)

    def test_two_qubit_trace(self):
        """Trace to two qubits should give 4x4 matrix."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        result = partial_trace_fast(rho, [0, 1], 4)
        assert result.shape == (4, 4)
        # Should be hermitian
        assert np.allclose(result, result.conj().T)
        # Trace should be 1
        assert np.abs(np.trace(result) - 1.0) < 1e-10

    def test_ghz_reduced_maximally_mixed(self):
        """Tracing GHZ to single qubit should give maximally mixed state."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        rho_1 = partial_trace_fast(rho, [0], 4)
        # Should be I/2 (maximally mixed)
        expected = np.eye(2) / 2
        assert np.allclose(rho_1, expected, atol=1e-10)


class TestVonNeumannEntropy:
    """Tests for von Neumann entropy computation."""

    def test_pure_state_entropy(self):
        """Pure state should have S = 0."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        S = von_neumann_entropy(rho)
        assert np.abs(S) < 1e-10

    def test_maximally_mixed_entropy(self):
        """Maximally mixed state should have S = log2(d)."""
        # 2x2 maximally mixed state
        rho = np.eye(2) / 2
        S = von_neumann_entropy(rho)
        # Should be log2(2) = 1
        assert np.abs(S - 1.0) < 1e-10

        # 4x4 maximally mixed state
        rho4 = np.eye(4) / 4
        S4 = von_neumann_entropy(rho4)
        # Should be log2(4) = 2
        assert np.abs(S4 - 2.0) < 1e-10

    def test_reduced_ghz_entropy(self):
        """Reduced density matrix of GHZ should have S = 1."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        rho_1 = partial_trace_fast(rho, [0], 4)
        S = von_neumann_entropy(rho_1)
        # Single qubit reduced state of GHZ is maximally mixed
        assert np.abs(S - 1.0) < 1e-10


class TestMutualInformation:
    """Tests for quantum mutual information."""

    def test_product_state_mi(self):
        """Product state should have MI = 0 for all pairs."""
        state = product_state(4)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 4)
        # All off-diagonal should be 0
        np.fill_diagonal(MI, 0)
        assert np.allclose(MI, 0, atol=1e-10)

    def test_ghz_mi(self):
        """GHZ state should have MI = 1 for all pairs."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 4)

        # Check all pairs
        for i in range(4):
            for j in range(i+1, 4):
                # GHZ has I(i:j) = 1 bit for all pairs
                assert np.abs(MI[i, j] - 1.0) < 0.1, f"MI[{i},{j}] = {MI[i,j]}"

    def test_mi_symmetry(self):
        """MI matrix should be symmetric."""
        state = w_state(4)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 4)
        assert np.allclose(MI, MI.T)

    def test_mi_nonnegative(self):
        """MI should be non-negative."""
        state = haar_random_state(4, seed=42)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 4)
        # Allow small numerical error
        assert np.all(MI >= -1e-10)

    def test_mi_bounded(self):
        """MI should be bounded by 2 * min(S_A, S_B)."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 4)
        # For qubits, max MI = 2 (both qubits maximally mixed)
        assert np.all(MI <= 2.1)  # Allow small numerical error


class TestMIToDistance:
    """Tests for MI to distance conversion."""

    def test_sqrt_method(self):
        """sqrt method should give valid distances."""
        MI = np.array([[0, 1, 0.5], [1, 0, 0.5], [0.5, 0.5, 0]])
        D = mi_to_distance(MI, method='sqrt')

        # Should be symmetric
        assert np.allclose(D, D.T)
        # Diagonal should be 0
        assert np.allclose(np.diag(D), 0)
        # Should be non-negative
        assert np.all(D >= 0)
        # Higher MI = lower distance
        assert D[0, 1] < D[0, 2]  # MI(0,1)=1 > MI(0,2)=0.5

    def test_inverse_method(self):
        """inverse method should give valid distances."""
        MI = np.array([[0, 1, 0.5], [1, 0, 0.5], [0.5, 0.5, 0]])
        D = mi_to_distance(MI, method='inverse')

        # Higher MI = lower distance
        assert D[0, 1] < D[0, 2]

    def test_negative_method(self):
        """negative method should give valid distances."""
        MI = np.array([[0, 1, 0.5], [1, 0, 0.5], [0.5, 0.5, 0]])
        D = mi_to_distance(MI, method='negative')

        # Higher MI = lower distance
        assert D[0, 1] < D[0, 2]


class TestValidateMIMatrix:
    """Tests for MI matrix validation."""

    def test_valid_matrix(self):
        """Valid MI matrix should pass all checks."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 4)
        result = validate_mi_matrix(MI)

        assert result['is_symmetric']
        assert result['is_nonnegative']
        assert result['diagonal_zero']
        assert result['max_reasonable']


class TestEntanglementEntropy:
    """Tests for entanglement entropy."""

    def test_product_state_entanglement(self):
        """Product state should have zero entanglement entropy."""
        state = product_state(4)
        rho = state_to_density_matrix(state)
        S_ent = half_chain_entanglement(rho, 4)
        assert np.abs(S_ent) < 1e-10

    def test_ghz_entanglement(self):
        """GHZ state should have S_ent = 1."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)
        S_ent = half_chain_entanglement(rho, 4)
        # Half-chain entanglement of GHZ is 1 bit
        assert np.abs(S_ent - 1.0) < 1e-10


# Import for random state test
from quantum_states import haar_random_state
