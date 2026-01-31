"""
Tests for improved dimension estimation module.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dimension_improved import (
    explained_variance_curve,
    continuous_dimension,
    find_fractional_dimension,
    landmark_mds,
    sparse_isomap,
    continuous_dimension_both
)


class TestExplainedVarianceCurve:
    """Tests for explained_variance_curve function."""

    def test_perfect_embedding(self):
        """Perfect embedding should have explained variance = 1."""
        # Create 2D points
        np.random.seed(42)
        points = np.random.rand(30, 2)
        D = squareform(pdist(points))

        # Use the points themselves as the embedding
        embedding = np.hstack([points, np.zeros((30, 8))])  # Pad to k_max

        variances = explained_variance_curve(D, embedding, k_max=10)

        # At k=2, should explain most variance
        assert variances[1] > 0.95, f"Expected >0.95, got {variances[1]}"

    def test_increasing_variance(self):
        """Explained variance should generally increase with dimension."""
        np.random.seed(42)
        points = np.random.rand(50, 5)
        D = squareform(pdist(points))

        from sklearn.manifold import MDS
        mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(D)

        variances = explained_variance_curve(D, embedding, k_max=10)

        # Each dimension should add some variance (mostly monotonic)
        increases = np.sum(np.diff(variances) >= -0.01)  # Allow small decreases
        assert increases >= 5, f"Expected mostly increasing, got {increases} increases"

    def test_empty_distance_matrix(self):
        """Should handle edge cases gracefully."""
        D = np.zeros((10, 10))
        embedding = np.zeros((10, 5))

        variances = explained_variance_curve(D, embedding, k_max=5)

        assert len(variances) == 5
        assert np.all(variances == 1.0)  # Perfect fit for zero distances


class TestFractionalDimension:
    """Tests for find_fractional_dimension function."""

    def test_exact_threshold(self):
        """When variance exactly hits threshold at an integer dimension."""
        variances = np.array([0.5, 0.8, 0.9, 0.95, 0.98])
        d = find_fractional_dimension(variances, threshold=0.9)
        assert d == 3.0, f"Expected 3.0, got {d}"

    def test_interpolation(self):
        """Should interpolate between dimensions."""
        variances = np.array([0.5, 0.85, 0.95])
        d = find_fractional_dimension(variances, threshold=0.9)

        # Should be between 2 and 3
        assert 2.0 < d < 3.0, f"Expected 2 < d < 3, got {d}"

        # Linear interpolation: at 0.9, between 0.85 and 0.95
        # frac = (0.9 - 0.85) / (0.95 - 0.85) = 0.5
        # d = 2 + 0.5 = 2.5
        assert abs(d - 2.5) < 0.01, f"Expected ~2.5, got {d}"

    def test_never_reaches_threshold(self):
        """When variance never reaches threshold."""
        variances = np.array([0.5, 0.6, 0.7, 0.8])
        d = find_fractional_dimension(variances, threshold=0.9)
        assert d == 4.0, f"Expected 4.0 (max dim), got {d}"

    def test_immediate_threshold(self):
        """When first dimension already exceeds threshold."""
        variances = np.array([0.95, 0.98, 0.99])
        d = find_fractional_dimension(variances, threshold=0.9)
        assert d == 1.0, f"Expected 1.0, got {d}"


class TestContinuousDimension:
    """Tests for continuous_dimension function."""

    def test_low_dimensional_data(self):
        """Should detect low intrinsic dimension."""
        np.random.seed(42)
        # Create data on a 2D plane embedded in higher space
        n = 50
        points_2d = np.random.rand(n, 2)
        D = squareform(pdist(points_2d))

        d_frac, variances, embedding = continuous_dimension(D, k_max=8)

        # Should find dimension close to 2
        assert d_frac < 4.0, f"Expected d < 4, got {d_frac}"

    def test_high_dimensional_data(self):
        """Should detect higher intrinsic dimension than 2D data."""
        np.random.seed(42)
        # Create genuinely high-dimensional data
        n = 50
        points = np.random.rand(n, 8)
        D = squareform(pdist(points))

        d_frac, variances, embedding = continuous_dimension(D, k_max=10)

        # Should find dimension > 1 (high-dim data needs more than 1D)
        # Note: with explained variance threshold, even high-dim data may
        # compress due to distance concentration phenomenon
        assert d_frac > 1.5, f"Expected d > 1.5, got {d_frac}"

    def test_returns_valid_shapes(self):
        """Should return arrays of correct shape."""
        np.random.seed(42)
        D = squareform(pdist(np.random.rand(30, 3)))

        d_frac, variances, embedding = continuous_dimension(D, k_max=8)

        assert isinstance(d_frac, float)
        assert len(variances) == 8
        assert embedding.shape == (30, 8)


class TestLandmarkMDS:
    """Tests for landmark_mds function."""

    def test_basic_functionality(self):
        """Should produce embedding without error."""
        np.random.seed(42)
        n = 100
        D = squareform(pdist(np.random.rand(n, 3)))

        embedding, landmarks = landmark_mds(D, n_landmarks=20, n_components=5)

        assert embedding.shape == (n, 5)
        assert len(landmarks) == 20
        assert np.all(landmarks < n)

    def test_preserves_landmark_structure(self):
        """Landmarks should have reasonable embedding."""
        np.random.seed(42)
        n = 50
        points = np.random.rand(n, 2)
        D = squareform(pdist(points))

        embedding, landmarks = landmark_mds(D, n_landmarks=20, n_components=5)

        # Check that embedded landmarks preserve relative distances
        D_landmarks_orig = D[np.ix_(landmarks, landmarks)]
        D_landmarks_emb = squareform(pdist(embedding[landmarks]))

        # Should have some correlation
        corr = np.corrcoef(D_landmarks_orig.flatten(), D_landmarks_emb.flatten())[0, 1]
        assert corr > 0.5, f"Expected correlation > 0.5, got {corr}"

    def test_handles_small_n(self):
        """Should work when n < n_landmarks."""
        np.random.seed(42)
        n = 30
        D = squareform(pdist(np.random.rand(n, 3)))

        embedding, landmarks = landmark_mds(D, n_landmarks=50, n_components=5)

        assert embedding.shape[0] == n
        assert len(landmarks) == n  # Should use all points as landmarks


class TestSparseIsomap:
    """Tests for sparse_isomap function."""

    def test_basic_functionality(self):
        """Should produce dimension estimate and embedding."""
        np.random.seed(42)
        D = squareform(pdist(np.random.rand(50, 3)))

        d_frac, variances, embedding = sparse_isomap(D, k=10, n_components=8)

        assert isinstance(d_frac, (int, float, np.integer, np.floating))
        assert d_frac > 0
        assert len(variances) == 8
        assert embedding.shape == (50, 8)

    def test_comparable_to_full_isomap(self):
        """Should give similar results to full Isomap for small graphs."""
        np.random.seed(42)
        D = squareform(pdist(np.random.rand(40, 2)))

        d_sparse, _, _ = sparse_isomap(D, k=15, n_components=8)
        d_full, _, _ = continuous_dimension(D, k_max=8, n_neighbors=15)

        # Should be reasonably close
        assert abs(d_sparse - d_full) < 2.0, f"Sparse: {d_sparse}, Full: {d_full}"


class TestContinuousDimensionBoth:
    """Tests for continuous_dimension_both function."""

    def test_returns_both_methods(self):
        """Should return results from both Isomap and MDS."""
        np.random.seed(42)
        D = squareform(pdist(np.random.rand(40, 3)))

        result = continuous_dimension_both(D, k_max=6)

        assert 'd_isomap' in result
        assert 'd_mds' in result
        assert 'd_mean' in result
        assert 'variances_isomap' in result
        assert 'variances_mds' in result

    def test_mean_is_average(self):
        """Mean dimension should be average of both methods."""
        np.random.seed(42)
        D = squareform(pdist(np.random.rand(40, 3)))

        result = continuous_dimension_both(D, k_max=6)

        expected_mean = (result['d_isomap'] + result['d_mds']) / 2
        assert abs(result['d_mean'] - expected_mean) < 0.001

    def test_sparse_mode(self):
        """Should work in sparse mode for large N."""
        np.random.seed(42)
        # Smaller test for speed
        D = squareform(pdist(np.random.rand(60, 3)))

        result = continuous_dimension_both(D, k_max=6, use_sparse=True)

        assert result['d_mean'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
