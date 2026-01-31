"""Tests for quantum state generation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_states import (
    product_state, ghz_state, w_state,
    cluster_state_1d, cluster_state_2d, apply_cz,
    haar_random_state, state_to_density_matrix,
    get_state_by_name, verify_state_normalization, verify_density_matrix
)


class TestProductState:
    """Tests for product state generation."""

    def test_normalization(self):
        """Product state should be normalized."""
        for n in [2, 4, 6]:
            state = product_state(n)
            assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_correct_form(self):
        """Product state should be |00...0>."""
        state = product_state(4)
        assert state[0] == 1.0
        assert np.allclose(state[1:], 0)

    def test_dimension(self):
        """Product state should have dimension 2^n."""
        for n in [2, 3, 4, 5]:
            state = product_state(n)
            assert len(state) == 2 ** n


class TestGHZState:
    """Tests for GHZ state generation."""

    def test_normalization(self):
        """GHZ state should be normalized."""
        for n in [2, 4, 6]:
            state = ghz_state(n)
            assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_correct_form(self):
        """GHZ state should be (|00...0> + |11...1>)/sqrt(2)."""
        state = ghz_state(4)
        expected_amp = 1.0 / np.sqrt(2)
        assert np.abs(state[0] - expected_amp) < 1e-10
        assert np.abs(state[-1] - expected_amp) < 1e-10
        # All other amplitudes should be 0
        other_indices = list(range(1, len(state) - 1))
        assert np.allclose(state[other_indices], 0)

    def test_two_qubit_bell(self):
        """2-qubit GHZ should be Bell state."""
        state = ghz_state(2)
        # |00> + |11> / sqrt(2)
        assert np.abs(state[0] - 1/np.sqrt(2)) < 1e-10  # |00>
        assert np.abs(state[1]) < 1e-10  # |01>
        assert np.abs(state[2]) < 1e-10  # |10>
        assert np.abs(state[3] - 1/np.sqrt(2)) < 1e-10  # |11>


class TestWState:
    """Tests for W state generation."""

    def test_normalization(self):
        """W state should be normalized."""
        for n in [2, 4, 6]:
            state = w_state(n)
            assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_correct_amplitudes(self):
        """W state amplitudes should be 1/sqrt(N)."""
        n = 4
        state = w_state(n)
        expected_amp = 1.0 / np.sqrt(n)

        # States with exactly one 1: indices 1, 2, 4, 8 for n=4
        one_hot_indices = [2**i for i in range(n)]
        for idx in one_hot_indices:
            assert np.abs(state[idx] - expected_amp) < 1e-10

    def test_three_qubit(self):
        """3-qubit W state: (|100> + |010> + |001>)/sqrt(3)."""
        state = w_state(3)
        expected = 1.0 / np.sqrt(3)
        # |001> = index 1, |010> = index 2, |100> = index 4
        assert np.abs(state[1] - expected) < 1e-10
        assert np.abs(state[2] - expected) < 1e-10
        assert np.abs(state[4] - expected) < 1e-10


class TestClusterState:
    """Tests for cluster state generation."""

    def test_1d_normalization(self):
        """1D cluster state should be normalized."""
        for n in [2, 4, 6]:
            state = cluster_state_1d(n)
            assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_2d_normalization(self):
        """2D cluster state should be normalized."""
        for rows, cols in [(2, 2), (2, 3), (3, 3)]:
            state = cluster_state_2d(rows, cols)
            assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_cz_gate(self):
        """CZ gate should flip phase when both qubits are 1."""
        # Start with |11> state
        state = np.array([0, 0, 0, 1], dtype=np.complex128)
        result = apply_cz(state, 0, 1, 2)
        # Should become -|11>
        assert np.abs(result[3] + 1.0) < 1e-10


class TestHaarRandomState:
    """Tests for Haar random state generation."""

    def test_normalization(self):
        """Random state should be normalized."""
        for n in [2, 4, 6]:
            state = haar_random_state(n, seed=42)
            assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_reproducibility(self):
        """Same seed should give same state."""
        state1 = haar_random_state(4, seed=123)
        state2 = haar_random_state(4, seed=123)
        assert np.allclose(state1, state2)

    def test_different_seeds(self):
        """Different seeds should give different states."""
        state1 = haar_random_state(4, seed=1)
        state2 = haar_random_state(4, seed=2)
        assert not np.allclose(state1, state2)


class TestDensityMatrix:
    """Tests for density matrix conversion."""

    def test_pure_state_density(self):
        """Density matrix of pure state should have Tr(rho^2) = 1."""
        state = ghz_state(4)
        rho = state_to_density_matrix(state)

        # Should be hermitian
        assert np.allclose(rho, rho.conj().T)

        # Trace should be 1
        assert np.abs(np.trace(rho) - 1.0) < 1e-10

        # Purity should be 1
        purity = np.real(np.trace(rho @ rho))
        assert np.abs(purity - 1.0) < 1e-10

    def test_verify_density_matrix(self):
        """Verification function should pass for valid density matrices."""
        state = w_state(4)
        rho = state_to_density_matrix(state)
        result = verify_density_matrix(rho)

        assert result['is_hermitian']
        assert result['trace_one']
        assert result['is_positive']
        assert result['is_pure']


class TestGetStateByName:
    """Tests for state factory function."""

    def test_all_state_types(self):
        """Should generate all supported state types."""
        n = 4
        for name in ['product', 'ghz', 'w', 'cluster', 'random']:
            state, metadata = get_state_by_name(name, n, seed=42)
            assert verify_state_normalization(state)
            assert metadata['name'] == name
            assert metadata['n_qubits'] == n

    def test_2d_cluster_geometry(self):
        """Should handle 2D cluster state geometry."""
        state, metadata = get_state_by_name('cluster', 6, geometry='2x3')
        assert verify_state_normalization(state)
        assert metadata['grid_shape'] == (2, 3)
        assert metadata['entanglement_type'] == 'topological_2d'

    def test_invalid_state_name(self):
        """Should raise for invalid state name."""
        with pytest.raises(ValueError):
            get_state_by_name('invalid', 4)
