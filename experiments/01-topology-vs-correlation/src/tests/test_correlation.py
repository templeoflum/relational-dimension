"""Tests for correlation module."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from correlation import (
    correlation_nn, correlation_lr, correlation_rand,
    correlation_to_distance, ensure_positive_semidefinite,
    count_correlations
)
from graph_generation import create_lattice, get_positions, get_adjacency


class TestCorrelationNN:
    """Tests for nearest-neighbor correlation."""

    def test_diagonal_is_one(self):
        """Self-correlation should be 1."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        C = correlation_nn(adj, 0.5)
        np.testing.assert_array_almost_equal(np.diag(C), 1.0)

    def test_neighbors_correlated(self):
        """Adjacent nodes should have correlation rho."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        C = correlation_nn(adj, 0.5)
        # After PSD projection, values may differ slightly
        assert C[0, 1] > 0.3  # Should be close to 0.5

    def test_non_neighbors_low_correlation(self):
        """Non-adjacent nodes should have low correlation."""
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        C = correlation_nn(adj, 0.5)
        # Nodes 0 and 2 are not adjacent
        assert abs(C[0, 2]) < 0.3

    def test_symmetric(self):
        """Correlation matrix should be symmetric."""
        G = create_lattice(5)
        adj = get_adjacency(G)
        C = correlation_nn(adj, 0.8)
        np.testing.assert_array_almost_equal(C, C.T)

    def test_positive_semidefinite(self):
        """Result should be positive semidefinite."""
        G = create_lattice(5)
        adj = get_adjacency(G)
        C = correlation_nn(adj, 0.8)
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues >= -1e-10)


class TestCorrelationLR:
    """Tests for long-range correlation."""

    def test_diagonal_is_one(self):
        """Self-correlation should be 1."""
        positions = np.random.rand(10, 2)
        C = correlation_lr(positions, 0.8)
        np.testing.assert_array_almost_equal(np.diag(C), 1.0)

    def test_close_nodes_high_correlation(self):
        """Nearby nodes should have high correlation."""
        positions = np.array([[0, 0], [0.01, 0], [0.5, 0.5]])
        C = correlation_lr(positions, 0.9, lambda_corr=0.1)
        assert C[0, 1] > C[0, 2]  # Closer nodes more correlated

    def test_symmetric(self):
        """Correlation matrix should be symmetric."""
        positions = np.random.rand(20, 2)
        C = correlation_lr(positions, 0.8)
        np.testing.assert_array_almost_equal(C, C.T)

    def test_positive_semidefinite(self):
        """Result should be positive semidefinite."""
        positions = np.random.rand(20, 2)
        C = correlation_lr(positions, 0.8)
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues >= -1e-10)


class TestCorrelationRand:
    """Tests for random correlation."""

    def test_diagonal_is_one(self):
        """Self-correlation should be 1."""
        C = correlation_rand(10, 5, 0.5, seed=42)
        np.testing.assert_array_almost_equal(np.diag(C), 1.0)

    def test_reproducible_with_seed(self):
        """Same seed should produce same matrix."""
        C1 = correlation_rand(20, 10, 0.5, seed=123)
        C2 = correlation_rand(20, 10, 0.5, seed=123)
        np.testing.assert_array_equal(C1, C2)

    def test_symmetric(self):
        """Correlation matrix should be symmetric."""
        C = correlation_rand(15, 8, 0.6, seed=42)
        np.testing.assert_array_almost_equal(C, C.T)

    def test_positive_semidefinite(self):
        """Result should be positive semidefinite."""
        C = correlation_rand(15, 8, 0.6, seed=42)
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues >= -1e-10)


class TestCorrelationToDistance:
    """Tests for correlation to distance conversion."""

    def test_perfect_correlation_zero_distance(self):
        """C=1 should give D=0."""
        C = np.array([[1, 1], [1, 1]])
        D = correlation_to_distance(C)
        np.testing.assert_array_almost_equal(D, 0)

    def test_no_correlation_sqrt2_distance(self):
        """C=0 should give D=sqrt(2)."""
        C = np.array([[1, 0], [0, 1]])
        D = correlation_to_distance(C)
        assert abs(D[0, 1] - np.sqrt(2)) < 1e-10

    def test_anticorrelation_distance_2(self):
        """C=-1 should give D=2."""
        C = np.array([[1, -1], [-1, 1]])
        D = correlation_to_distance(C)
        assert abs(D[0, 1] - 2) < 1e-10

    def test_diagonal_is_zero(self):
        """Distance to self should be 0."""
        C = np.random.rand(5, 5)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 1)
        D = correlation_to_distance(C)
        np.testing.assert_array_almost_equal(np.diag(D), 0)


class TestEnsurePositiveSemidefinite:
    """Tests for PSD projection."""

    def test_already_psd_unchanged(self):
        """PSD matrix should be (nearly) unchanged."""
        C = np.eye(5)
        C_psd = ensure_positive_semidefinite(C)
        np.testing.assert_array_almost_equal(C, C_psd)

    def test_not_psd_becomes_psd(self):
        """Non-PSD matrix should become PSD."""
        # Create a matrix that's not PSD
        C = np.array([[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]])
        C[0, 2] = 0.1  # Make it inconsistent
        C[2, 0] = 0.1

        C_psd = ensure_positive_semidefinite(C)
        eigenvalues = np.linalg.eigvalsh(C_psd)
        assert np.all(eigenvalues >= -1e-10)

    def test_diagonal_preserved(self):
        """Diagonal should be 1."""
        C = np.random.rand(5, 5)
        C = (C + C.T) / 2
        C_psd = ensure_positive_semidefinite(C)
        np.testing.assert_array_almost_equal(np.diag(C_psd), 1.0)


class TestCountCorrelations:
    """Tests for correlation counting."""

    def test_identity_zero_correlations(self):
        """Identity matrix should have 0 significant correlations."""
        C = np.eye(5)
        assert count_correlations(C, threshold=0.1) == 0

    def test_counts_above_threshold(self):
        """Should count pairs above threshold."""
        C = np.array([[1, 0.5, 0.05], [0.5, 1, 0.2], [0.05, 0.2, 1]])
        assert count_correlations(C, threshold=0.1) == 2  # (0,1) and (1,2)
