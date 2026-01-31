"""Tests for lattice generation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lattice_generation import (
    generate_1d_chain,
    generate_2d_lattice,
    generate_3d_lattice,
    distances_to_correlation,
    generate_calibration_system
)


class TestGenerate1DChain:
    """Tests for 1D chain generation."""

    def test_output_shapes(self):
        """Check output array shapes."""
        n = 50
        positions, distances, metadata = generate_1d_chain(n)

        assert positions.shape == (n, 1)
        assert distances.shape == (n, n)
        assert metadata['n_nodes'] == n
        assert metadata['true_dimension'] == 1

    def test_distance_properties(self):
        """Check distance matrix properties."""
        _, D, _ = generate_1d_chain(20)

        # Symmetric
        assert np.allclose(D, D.T)

        # Non-negative
        assert np.all(D >= 0)

        # Zero diagonal
        assert np.allclose(np.diag(D), 0)

    def test_distances_correct(self):
        """Check distance values are correct."""
        _, D, _ = generate_1d_chain(5)

        # Distance between adjacent nodes should be 1
        assert np.isclose(D[0, 1], 1.0)
        assert np.isclose(D[1, 2], 1.0)

        # Distance between nodes 0 and 4 should be 4
        assert np.isclose(D[0, 4], 4.0)


class TestGenerate2DLattice:
    """Tests for 2D lattice generation."""

    def test_output_shapes(self):
        """Check output array shapes."""
        side = 5
        positions, distances, metadata = generate_2d_lattice(side)

        n = side * side
        assert positions.shape == (n, 2)
        assert distances.shape == (n, n)
        assert metadata['n_nodes'] == n
        assert metadata['true_dimension'] == 2

    def test_distance_properties(self):
        """Check distance matrix properties."""
        _, D, _ = generate_2d_lattice(7)

        assert np.allclose(D, D.T)
        assert np.all(D >= 0)
        assert np.allclose(np.diag(D), 0)

    def test_neighbor_distances(self):
        """Check adjacent node distances."""
        side = 5
        _, D, _ = generate_2d_lattice(side)

        # Horizontal neighbor: nodes 0 and 1 should be distance 1
        assert np.isclose(D[0, 1], 1.0)

        # Vertical neighbor: nodes 0 and 5 should be distance 1
        assert np.isclose(D[0, side], 1.0)

        # Diagonal: nodes 0 and 6 should be sqrt(2)
        assert np.isclose(D[0, side + 1], np.sqrt(2))


class TestGenerate3DLattice:
    """Tests for 3D lattice generation."""

    def test_output_shapes(self):
        """Check output array shapes."""
        dims = (3, 4, 5)
        positions, distances, metadata = generate_3d_lattice(dims)

        n = 3 * 4 * 5
        assert positions.shape == (n, 3)
        assert distances.shape == (n, n)
        assert metadata['n_nodes'] == n
        assert metadata['true_dimension'] == 3

    def test_distance_properties(self):
        """Check distance matrix properties."""
        _, D, _ = generate_3d_lattice((3, 3, 3))

        assert np.allclose(D, D.T)
        assert np.all(D >= 0)
        assert np.allclose(np.diag(D), 0)


class TestDistancesToCorrelation:
    """Tests for distance to correlation conversion."""

    def test_output_properties(self):
        """Check correlation matrix properties."""
        _, D, _ = generate_2d_lattice(5)
        C = distances_to_correlation(D)

        # Symmetric
        assert np.allclose(C, C.T)

        # Diagonal is 1
        assert np.allclose(np.diag(C), 1.0)

        # Values in valid range
        assert np.all(C >= -1 - 1e-10)
        assert np.all(C <= 1 + 1e-10)

    def test_correlation_monotonicity(self):
        """Closer nodes should have higher correlation."""
        _, D, _ = generate_1d_chain(10)
        C = distances_to_correlation(D)

        # Node 0 should be more correlated with 1 than with 5
        assert C[0, 1] > C[0, 5]


class TestGenerateCalibrationSystem:
    """Tests for calibration system generation."""

    def test_1d_system(self):
        """Test 1D calibration system."""
        system = generate_calibration_system('1d', 50)

        assert 'positions' in system
        assert 'D_topo' in system
        assert 'C' in system
        assert system['metadata']['true_dimension'] == 1

    def test_2d_system(self):
        """Test 2D calibration system."""
        system = generate_calibration_system('2d', 100)

        assert system['metadata']['true_dimension'] == 2
        # Should be approximately 10x10
        assert system['metadata']['n_nodes'] == 100

    def test_3d_system(self):
        """Test 3D calibration system."""
        system = generate_calibration_system('3d', 125)

        assert system['metadata']['true_dimension'] == 3

    def test_invalid_type(self):
        """Test invalid system type raises error."""
        with pytest.raises(ValueError):
            generate_calibration_system('4d', 100)
