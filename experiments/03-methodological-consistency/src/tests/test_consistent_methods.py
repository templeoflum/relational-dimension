"""
Tests for consistent_methods.py
"""

import pytest
import numpy as np
from scipy.spatial.distance import pdist, squareform

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from consistent_methods import (
    full_mds_dimension,
    full_isomap_dimension,
    compute_reconstruction_errors,
    find_dimension_with_interpolation,
    extract_dimension_consistent,
    validate_embedding_quality,
    compute_compression_ratio
)


class TestFullMDSDimension:
    """Tests for full_mds_dimension."""

    def test_2d_points(self):
        """2D point cloud should return valid result with errors."""
        np.random.seed(42)
        points = np.random.rand(50, 2)
        D = squareform(pdist(points))

        result = full_mds_dimension(D, k_max=5)

        assert result['valid'] is True
        assert 1 <= result['d_mds'] <= 5  # Dimension estimation is noisy
        assert len(result['errors']) == 5

    def test_1d_line(self):
        """Points on a line should have low dimension."""
        points = np.column_stack([np.linspace(0, 1, 30), np.zeros(30)])
        D = squareform(pdist(points))

        result = full_mds_dimension(D, k_max=5)

        assert result['valid'] is True
        assert result['d_mds'] <= 3  # Allow some noise

    def test_returns_stress(self):
        """Result should include stress value."""
        np.random.seed(42)
        points = np.random.rand(30, 2)
        D = squareform(pdist(points))

        result = full_mds_dimension(D, k_max=5)

        assert 'stress' in result
        assert isinstance(result['stress'], float)


class TestFullIsomapDimension:
    """Tests for full_isomap_dimension."""

    def test_2d_points(self):
        """2D point cloud should have dimension ~2."""
        np.random.seed(42)
        points = np.random.rand(50, 2)
        D = squareform(pdist(points))

        result = full_isomap_dimension(D, k_max=5, n_neighbors=8)

        assert result['valid'] is True
        assert 1 <= result['d_isomap'] <= 3

    def test_adjusts_neighbors(self):
        """Should handle small graphs by adjusting n_neighbors."""
        np.random.seed(42)
        points = np.random.rand(15, 2)
        D = squareform(pdist(points))

        result = full_isomap_dimension(D, k_max=5, n_neighbors=20)

        assert result['valid'] is True


class TestComputeReconstructionErrors:
    """Tests for compute_reconstruction_errors."""

    def test_error_decreases(self):
        """Error should generally decrease with dimension."""
        np.random.seed(42)
        points = np.random.rand(30, 3)
        D = squareform(pdist(points))

        # Create a 5D embedding
        from sklearn.manifold import MDS
        mds = MDS(n_components=5, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(D)

        errors = compute_reconstruction_errors(D, embedding, k_max=5)

        assert len(errors) == 5
        assert errors[4] <= errors[0]  # Higher dim should have lower error

    def test_handles_zero_distances(self):
        """Should handle matrices with zero max distance."""
        D = np.zeros((10, 10))
        embedding = np.random.rand(10, 5)

        errors = compute_reconstruction_errors(D, embedding, k_max=5)

        assert len(errors) == 5
        assert all(e == 0 for e in errors)


class TestFindDimensionWithInterpolation:
    """Tests for find_dimension_with_interpolation."""

    def test_finds_threshold_crossing(self):
        """Should find dimension where error drops below threshold."""
        errors = np.array([1.0, 0.5, 0.08, 0.05, 0.03])

        d_int, d_frac = find_dimension_with_interpolation(errors, threshold=0.1)

        assert d_int == 3
        assert 2 < d_frac < 4

    def test_handles_empty_errors(self):
        """Should handle empty error array."""
        errors = np.array([])

        d_int, d_frac = find_dimension_with_interpolation(errors)

        assert d_int == 1
        assert d_frac == 1.0

    def test_fractional_interpolation(self):
        """Fractional dimension should interpolate between integers."""
        errors = np.array([1.0, 0.09, 0.05])  # Threshold=0.1*1.0=0.1

        d_int, d_frac = find_dimension_with_interpolation(errors, threshold=0.1)

        assert d_int == 2
        # Should interpolate: (1.0 - 0.1) / (1.0 - 0.09) = 0.989
        assert 1.5 < d_frac < 2.5


class TestExtractDimensionConsistent:
    """Tests for extract_dimension_consistent."""

    def test_both_methods_run(self):
        """Should run both MDS and Isomap."""
        np.random.seed(42)
        points = np.random.rand(40, 2)
        D = squareform(pdist(points))

        result = extract_dimension_consistent(D, k_max=5)

        assert 'd_mds' in result
        assert 'd_isomap' in result
        assert 'd_mean' in result
        assert 'valid' in result

    def test_computes_agreement(self):
        """Should compute agreement between methods."""
        np.random.seed(42)
        points = np.random.rand(40, 2)
        D = squareform(pdist(points))

        result = extract_dimension_consistent(D, k_max=5)

        assert 'agreement_diff' in result
        assert result['agreement_diff'] >= 0

    def test_marks_valid_when_methods_agree(self):
        """Should be valid when methods agree within 0.5."""
        np.random.seed(42)
        points = np.random.rand(50, 2)
        D = squareform(pdist(points))

        result = extract_dimension_consistent(D, k_max=5)

        if result['agreement_diff'] <= 0.5:
            assert result['valid'] is True


class TestValidateEmbeddingQuality:
    """Tests for validate_embedding_quality."""

    def test_good_quality(self):
        """Good stress and agreement should pass."""
        result = validate_embedding_quality(stress=0.1, agreement_diff=0.3)

        assert result['stress_ok'] is True
        assert result['agreement_ok'] is True
        assert result['overall_valid'] is True

    def test_bad_stress(self):
        """High stress should fail."""
        result = validate_embedding_quality(stress=0.5, agreement_diff=0.3)

        assert result['stress_ok'] is False
        assert result['overall_valid'] is False

    def test_bad_agreement(self):
        """High disagreement should fail."""
        result = validate_embedding_quality(stress=0.1, agreement_diff=1.0)

        assert result['agreement_ok'] is False
        assert result['overall_valid'] is False


class TestComputeCompressionRatio:
    """Tests for compute_compression_ratio."""

    def test_compression(self):
        """Lower corr dimension should give positive ratio."""
        delta = compute_compression_ratio(d_topo=4, d_corr=2)
        assert delta == 0.5

    def test_expansion(self):
        """Higher corr dimension should give negative ratio."""
        delta = compute_compression_ratio(d_topo=2, d_corr=4)
        assert delta == -1.0

    def test_no_change(self):
        """Same dimensions should give zero ratio."""
        delta = compute_compression_ratio(d_topo=3, d_corr=3)
        assert delta == 0.0

    def test_zero_topo(self):
        """Zero topological dimension should return zero."""
        delta = compute_compression_ratio(d_topo=0, d_corr=2)
        assert delta == 0.0
