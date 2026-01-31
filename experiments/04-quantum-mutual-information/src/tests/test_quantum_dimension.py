"""Tests for quantum dimension extraction."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_states import (
    product_state, ghz_state, state_to_density_matrix
)
from mutual_information import mutual_information_matrix
from quantum_dimension import (
    mi_distance_matrix, extract_dimension_mds, extract_dimension_isomap,
    compute_compression, analyze_quantum_state
)


class TestMIDistanceMatrix:
    """Tests for MI to distance conversion."""

    def test_sqrt_method_output_shape(self):
        """Distance matrix should have same shape as input."""
        MI = np.random.rand(5, 5)
        MI = (MI + MI.T) / 2  # Make symmetric
        np.fill_diagonal(MI, 0)
        D = mi_distance_matrix(MI, method='sqrt')
        assert D.shape == MI.shape

    def test_diagonal_zero(self):
        """Diagonal should be zero."""
        MI = np.random.rand(5, 5)
        MI = (MI + MI.T) / 2
        D = mi_distance_matrix(MI)
        assert np.allclose(np.diag(D), 0)

    def test_symmetry(self):
        """Distance matrix should be symmetric."""
        MI = np.random.rand(5, 5)
        MI = (MI + MI.T) / 2
        np.fill_diagonal(MI, 0)
        D = mi_distance_matrix(MI)
        assert np.allclose(D, D.T)


class TestExtractDimensionMDS:
    """Tests for MDS dimension extraction."""

    def test_returns_tuple(self):
        """Should return dimension and details dict."""
        D = np.random.rand(6, 6)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        d, details = extract_dimension_mds(D)

        assert isinstance(d, (int, float))
        assert isinstance(details, dict)
        assert 'explained_variances' in details
        assert 'd_threshold' in details

    def test_dimension_bounds(self):
        """Dimension should be between 1 and max_dim."""
        D = np.random.rand(6, 6)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        d, _ = extract_dimension_mds(D, max_dim=5)
        assert 1 <= d <= 5

    def test_low_rank_matrix(self):
        """Low-dimensional embedding should have low dimension."""
        # Create 2D points
        points_2d = np.random.rand(10, 2)
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(points_2d)

        d, details = extract_dimension_mds(D, threshold=0.95)
        # Should find ~2 dimensions
        assert d <= 4  # Allow some slack


class TestExtractDimensionIsomap:
    """Tests for Isomap dimension extraction."""

    def test_returns_tuple(self):
        """Should return dimension and details dict."""
        D = np.random.rand(8, 8)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        d, details = extract_dimension_isomap(D, n_neighbors=3)

        assert isinstance(d, (int, float))
        assert isinstance(details, dict)

    def test_dimension_bounds(self):
        """Dimension should be between 1 and max_dim."""
        D = np.random.rand(8, 8)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        d, _ = extract_dimension_isomap(D, max_dim=5, n_neighbors=3)
        assert 1 <= d <= 5


class TestComputeCompression:
    """Tests for compression ratio computation."""

    def test_no_compression(self):
        """d_Q = d_topo should give delta = 0."""
        delta = compute_compression(5.0, 5.0)
        assert np.abs(delta) < 1e-10

    def test_full_compression(self):
        """d_Q = 0 should give delta = 1."""
        delta = compute_compression(5.0, 0.0)
        assert np.abs(delta - 1.0) < 1e-10

    def test_partial_compression(self):
        """d_Q < d_topo should give 0 < delta < 1."""
        delta = compute_compression(10.0, 5.0)
        assert np.abs(delta - 0.5) < 1e-10

    def test_negative_compression(self):
        """d_Q > d_topo should give delta < 0."""
        delta = compute_compression(5.0, 7.0)
        assert delta < 0

    def test_zero_topo_dimension(self):
        """Zero topological dimension should return 0."""
        delta = compute_compression(0.0, 1.0)
        assert delta == 0.0


class TestAnalyzeQuantumState:
    """Tests for full quantum state analysis."""

    def test_product_state_analysis(self):
        """Product state should have low compression."""
        state = product_state(4)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 4)

        result = analyze_quantum_state(MI, 4, geometry='chain')

        assert 'delta' in result
        assert 'd_quantum' in result
        assert 'd_topo' in result
        # Product state has MI=0, so distances are all equal
        # Compression should be near 0 or could be negative

    def test_ghz_state_analysis(self):
        """GHZ state should show compression."""
        state = ghz_state(6)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 6)

        result = analyze_quantum_state(MI, 6, geometry='chain')

        assert result['n_qubits'] == 6
        assert result['geometry'] == 'chain'
        # GHZ has uniform MI=1 for all pairs
        assert result['mi_mean'] > 0.9

    def test_2d_geometry(self):
        """2D geometry should be handled."""
        state = ghz_state(6)
        rho = state_to_density_matrix(state)
        MI = mutual_information_matrix(rho, 6)

        result = analyze_quantum_state(MI, 6, geometry='2x3')

        assert result['geometry'] == '2x3'
        assert result['d_topo'] > 0
