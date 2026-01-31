"""Tests for distance transformations."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distance_transforms import (
    transform_sqrt,
    transform_linear,
    transform_arccos,
    transform_neglog,
    get_transform,
    list_transforms,
    apply_transform,
    validate_distance_matrix
)


class TestTransformSqrt:
    """Tests for sqrt transformation."""

    def test_perfect_correlation(self):
        """C=1 should give D=0."""
        C = np.array([[1, 1], [1, 1]])
        D = transform_sqrt(C)
        assert np.allclose(D, 0)

    def test_zero_correlation(self):
        """C=0 should give D=sqrt(2)."""
        C = np.array([[1, 0], [0, 1]])
        D = transform_sqrt(C)
        assert np.isclose(D[0, 1], np.sqrt(2))

    def test_symmetric_output(self):
        """Output should be symmetric."""
        C = np.random.rand(5, 5)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 1)
        D = transform_sqrt(C)
        assert np.allclose(D, D.T)


class TestTransformLinear:
    """Tests for linear transformation."""

    def test_perfect_correlation(self):
        """C=1 should give D=0."""
        C = np.array([[1, 1], [1, 1]])
        D = transform_linear(C)
        assert np.allclose(D, 0)

    def test_zero_correlation(self):
        """C=0 should give D=1."""
        C = np.array([[1, 0], [0, 1]])
        D = transform_linear(C)
        assert np.isclose(D[0, 1], 1.0)


class TestTransformArccos:
    """Tests for arccos transformation."""

    def test_perfect_correlation(self):
        """C=1 should give D=0."""
        C = np.array([[1, 1], [1, 1]])
        D = transform_arccos(C)
        assert np.allclose(D, 0)

    def test_zero_correlation(self):
        """C=0 should give D=pi/2."""
        C = np.array([[1, 0], [0, 1]])
        D = transform_arccos(C)
        assert np.isclose(D[0, 1], np.pi / 2)


class TestTransformNeglog:
    """Tests for negative log transformation."""

    def test_high_correlation_low_distance(self):
        """Higher correlation should give lower distance."""
        C = np.array([[1, 0.9, 0.1], [0.9, 1, 0.1], [0.1, 0.1, 1]])
        D = transform_neglog(C)

        # C[0,1]=0.9 should give lower D than C[0,2]=0.1
        assert D[0, 1] < D[0, 2]


class TestTransformRegistry:
    """Tests for transform registry functions."""

    def test_list_transforms(self):
        """Should list available transforms."""
        transforms = list_transforms()
        assert 'sqrt' in transforms
        assert 'linear' in transforms
        assert 'arccos' in transforms
        assert 'neglog' in transforms

    def test_get_transform(self):
        """Should get transform function by name."""
        fn = get_transform('sqrt')
        assert callable(fn)

    def test_get_unknown_transform(self):
        """Should raise for unknown transform."""
        with pytest.raises(ValueError):
            get_transform('unknown')

    def test_apply_transform(self):
        """Should apply named transform."""
        C = np.array([[1, 0.5], [0.5, 1]])
        D = apply_transform(C, 'sqrt')
        assert D.shape == C.shape


class TestValidateDistanceMatrix:
    """Tests for distance matrix validation."""

    def test_valid_matrix(self):
        """Valid distance matrix should pass all checks."""
        # Create valid distance matrix from positions
        pos = np.random.rand(10, 2)
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(pos))

        result = validate_distance_matrix(D)
        assert result['is_symmetric']
        assert result['is_nonnegative']
        assert result['diagonal_zero']
        assert result['is_valid_metric']

    def test_invalid_asymmetric(self):
        """Asymmetric matrix should fail."""
        D = np.array([[0, 1], [2, 0]])
        result = validate_distance_matrix(D)
        assert not result['is_symmetric']

    def test_invalid_negative(self):
        """Negative values should fail."""
        D = np.array([[0, -1], [-1, 0]])
        result = validate_distance_matrix(D)
        assert not result['is_nonnegative']

    def test_invalid_diagonal(self):
        """Non-zero diagonal should fail."""
        D = np.array([[1, 1], [1, 1]])
        result = validate_distance_matrix(D)
        assert not result['diagonal_zero']
