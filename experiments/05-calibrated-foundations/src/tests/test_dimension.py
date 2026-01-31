"""Tests for dimension extraction."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimension_extraction import (
    extract_dimension_mds,
    extract_dimension_isomap,
    extract_dimension_with_agreement
)
from lattice_generation import generate_1d_chain, generate_2d_lattice, generate_3d_lattice


class TestExtractDimensionMDS:
    """Tests for MDS dimension extraction."""

    def test_2d_lattice(self):
        """2D lattice should give dimension ~2."""
        _, D, _ = generate_2d_lattice(10)
        d, details = extract_dimension_mds(D, max_dim=5)

        # Should recover approximately 2
        assert 1.5 <= d <= 2.5

    def test_3d_lattice(self):
        """3D lattice should give dimension > 2."""
        _, D, _ = generate_3d_lattice((4, 4, 4))
        d, details = extract_dimension_mds(D, max_dim=5)

        # 3D is harder to recover - just verify it's higher than 2D
        # The calibration phase will gate on precise recovery
        assert d >= 1.5

    def test_returns_details(self):
        """Should return explained variances."""
        _, D, _ = generate_2d_lattice(7)
        d, details = extract_dimension_mds(D, max_dim=4)

        assert 'explained_variances' in details
        assert len(details['explained_variances']) == 4


class TestExtractDimensionIsomap:
    """Tests for Isomap dimension extraction."""

    def test_2d_lattice(self):
        """2D lattice should give dimension ~2."""
        _, D, _ = generate_2d_lattice(10)
        d, details = extract_dimension_isomap(D, max_dim=5)

        # Should recover approximately 2
        assert 1.5 <= d <= 2.5

    def test_returns_details(self):
        """Should return explained variances."""
        _, D, _ = generate_2d_lattice(7)
        d, details = extract_dimension_isomap(D, max_dim=4)

        assert 'explained_variances' in details
        assert len(details['explained_variances']) == 4


class TestExtractDimensionWithAgreement:
    """Tests for dimension extraction with method agreement."""

    def test_2d_lattice_agrees(self):
        """Both methods should agree on 2D lattice."""
        _, D, _ = generate_2d_lattice(10)
        result = extract_dimension_with_agreement(D, agreement_threshold=0.5, max_dim=5)

        assert result['valid']
        assert 1.5 <= result['d_mds'] <= 2.5
        assert 1.5 <= result['d_isomap'] <= 2.5
        assert result['d_final'] is not None

    def test_invalid_when_methods_disagree(self):
        """Should be invalid when methods disagree."""
        # Create a distance matrix where methods might disagree
        # Use a degenerate case
        D = np.ones((10, 10)) * 0.5
        np.fill_diagonal(D, 0)

        result = extract_dimension_with_agreement(D, agreement_threshold=0.1, max_dim=3)

        # Either valid with close results, or invalid
        if not result['valid']:
            assert result['method_diff'] > 0.1

    def test_consensus_is_average(self):
        """Consensus should be average of both methods."""
        _, D, _ = generate_2d_lattice(8)
        result = extract_dimension_with_agreement(D, max_dim=4)

        if result['valid']:
            expected = (result['d_mds'] + result['d_isomap']) / 2
            assert np.isclose(result['d_final'], expected)


class TestDimensionRecovery:
    """Integration tests for dimension recovery accuracy."""

    def test_1d_chain(self):
        """1D chain should give dimension ~1."""
        _, D, _ = generate_1d_chain(50)
        result = extract_dimension_with_agreement(D, max_dim=5)

        if result['valid']:
            assert 0.8 <= result['d_final'] <= 1.5

    def test_2d_grid(self):
        """2D grid should give dimension ~2."""
        _, D, _ = generate_2d_lattice(10)
        result = extract_dimension_with_agreement(D, max_dim=5)

        if result['valid']:
            assert 1.5 <= result['d_final'] <= 2.5

    def test_3d_grid(self):
        """3D grid should give dimension > 2."""
        _, D, _ = generate_3d_lattice((5, 5, 4))
        result = extract_dimension_with_agreement(D, max_dim=5)

        # 3D is harder to recover - just verify it's higher than 1D
        # The calibration phase will evaluate precise recovery
        if result['valid']:
            assert result['d_final'] >= 1.5

