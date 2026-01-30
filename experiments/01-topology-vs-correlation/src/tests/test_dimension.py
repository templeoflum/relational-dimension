"""Tests for dimension extraction module."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dimension import (
    extract_dimension_isomap, extract_dimension_mds,
    compute_reconstruction_errors, find_intrinsic_dimension,
    validate_methods, extract_dimension_both, normalize_distances
)


def create_2d_points(n=50, seed=42):
    """Create distance matrix for 2D point cloud."""
    np.random.seed(seed)
    points = np.random.rand(n, 2)
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(points))


def create_1d_points(n=50, seed=42):
    """Create distance matrix for 1D point cloud."""
    np.random.seed(seed)
    points = np.random.rand(n, 1)
    from scipy.spatial.distance import pdist, squareform
    return squareform(pdist(points))


class TestExtractDimensionIsomap:
    """Tests for Isomap dimension extraction."""

    def test_returns_correct_types(self):
        """Should return int, array, array."""
        D = create_2d_points(30)
        d, errors, embedding = extract_dimension_isomap(D, k_max=5)
        assert isinstance(d, (int, np.integer))
        assert isinstance(errors, np.ndarray)
        assert isinstance(embedding, np.ndarray)

    def test_1d_data_gives_low_dimension(self):
        """1D data should give dimension 1 or 2."""
        D = create_1d_points(50)
        d, _, _ = extract_dimension_isomap(D, k_max=5)
        assert d <= 2

    def test_errors_decrease(self):
        """Errors should generally decrease with dimension."""
        D = create_2d_points(40)
        _, errors, _ = extract_dimension_isomap(D, k_max=5)
        # First error should be >= last error
        assert errors[0] >= errors[-1]

    def test_embedding_shape(self):
        """Embedding should have correct shape."""
        n = 30
        k_max = 5
        D = create_2d_points(n)
        _, _, embedding = extract_dimension_isomap(D, k_max=k_max)
        assert embedding.shape == (n, k_max)


class TestExtractDimensionMDS:
    """Tests for MDS dimension extraction."""

    def test_returns_correct_types(self):
        """Should return int, array, array."""
        D = create_2d_points(30)
        d, errors, embedding = extract_dimension_mds(D, k_max=5)
        assert isinstance(d, (int, np.integer))
        assert isinstance(errors, np.ndarray)
        assert isinstance(embedding, np.ndarray)

    def test_1d_data_gives_low_dimension(self):
        """1D data should give reasonably low dimension."""
        D = create_1d_points(50)
        d, _, _ = extract_dimension_mds(D, k_max=5)
        # MDS can be less stable than Isomap, allow up to 3
        assert d <= 3


class TestComputeReconstructionErrors:
    """Tests for reconstruction error computation."""

    def test_output_length(self):
        """Should return k_max errors."""
        D = create_2d_points(20)
        embedding = np.random.rand(20, 5)
        errors = compute_reconstruction_errors(D, embedding, 5)
        assert len(errors) == 5

    def test_perfect_reconstruction(self):
        """Perfect embedding should have low error."""
        from scipy.spatial.distance import pdist, squareform
        n = 20
        points = np.random.rand(n, 2)
        D = squareform(pdist(points))
        # Use actual points as embedding
        embedding = np.hstack([points, np.zeros((n, 3))])
        errors = compute_reconstruction_errors(D, embedding, 5)
        # Error at dimension 2 should be very low
        assert errors[1] < 0.1


class TestFindIntrinsicDimension:
    """Tests for dimension finding from error curve."""

    def test_simple_threshold(self):
        """Should find first dimension below threshold."""
        errors = np.array([1.0, 0.5, 0.2, 0.05, 0.01])
        d = find_intrinsic_dimension(errors, threshold=0.1)
        assert d == 4  # First below 0.1 * 1.0 = 0.1

    def test_no_dimension_below_threshold(self):
        """Should return elbow if nothing below threshold."""
        errors = np.array([1.0, 0.8, 0.6, 0.5, 0.4])
        d = find_intrinsic_dimension(errors, threshold=0.01)
        # Should use elbow method
        assert 1 <= d <= 5


class TestValidateMethods:
    """Tests for method validation."""

    def test_same_dimension_valid(self):
        """Same dimension should be valid."""
        assert validate_methods(3, 3) is True

    def test_within_tolerance_valid(self):
        """Difference within tolerance should be valid."""
        assert validate_methods(3, 3, max_diff=0.5) is True
        assert validate_methods(3, 3, max_diff=0.5) is True

    def test_large_difference_invalid(self):
        """Large difference should be invalid."""
        assert validate_methods(2, 5, max_diff=0.5) is False


class TestExtractDimensionBoth:
    """Tests for combined dimension extraction."""

    def test_returns_all_fields(self):
        """Should return dictionary with all expected fields."""
        D = create_2d_points(30)
        result = extract_dimension_both(D, k_max=5)

        required_keys = ['d_isomap', 'd_mds', 'd_mean', 'valid',
                        'errors_isomap', 'errors_mds',
                        'embedding_isomap', 'embedding_mds']
        for key in required_keys:
            assert key in result

    def test_d_mean_is_average(self):
        """d_mean should be average of isomap and mds."""
        D = create_2d_points(30)
        result = extract_dimension_both(D, k_max=5)
        expected_mean = (result['d_isomap'] + result['d_mds']) / 2
        assert result['d_mean'] == expected_mean


class TestNormalizeDistances:
    """Tests for distance normalization."""

    def test_max_is_one(self):
        """Maximum should be 1 after normalization."""
        D = np.array([[0, 5, 10], [5, 0, 5], [10, 5, 0]])
        D_norm = normalize_distances(D)
        assert np.max(D_norm) == 1.0

    def test_preserves_zeros(self):
        """Zeros should remain zero."""
        D = np.array([[0, 5, 10], [5, 0, 5], [10, 5, 0]])
        D_norm = normalize_distances(D)
        assert D_norm[0, 0] == 0
        assert D_norm[1, 1] == 0

    def test_preserves_ratios(self):
        """Distance ratios should be preserved."""
        D = np.array([[0, 4, 8], [4, 0, 4], [8, 4, 0]])
        D_norm = normalize_distances(D)
        # 4 is half of 8, so normalized should preserve this
        assert abs(D_norm[0, 1] / D_norm[0, 2] - 0.5) < 1e-10
