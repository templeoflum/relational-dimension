"""Tests for graph generation module."""

import pytest
import numpy as np
import networkx as nx
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_generation import (
    create_rgg, create_lattice, get_positions, get_adjacency,
    get_graph_distances, get_largest_component, estimate_radius
)


class TestCreateRGG:
    """Tests for random geometric graph creation."""

    def test_correct_node_count(self):
        """RGG should have specified number of nodes."""
        G = create_rgg(50, 0.3, seed=42)
        assert G.number_of_nodes() == 50

    def test_positions_stored(self):
        """All nodes should have position attributes."""
        G = create_rgg(20, 0.3, seed=42)
        for i in range(20):
            assert 'pos' in G.nodes[i]
            pos = G.nodes[i]['pos']
            assert len(pos) == 2
            assert 0 <= pos[0] <= 1
            assert 0 <= pos[1] <= 1

    def test_reproducible_with_seed(self):
        """Same seed should produce same graph."""
        G1 = create_rgg(30, 0.3, seed=123)
        G2 = create_rgg(30, 0.3, seed=123)
        assert G1.number_of_edges() == G2.number_of_edges()

    def test_larger_radius_more_edges(self):
        """Larger radius should create more edges."""
        G_small = create_rgg(50, 0.2, seed=42)
        G_large = create_rgg(50, 0.4, seed=42)
        assert G_large.number_of_edges() >= G_small.number_of_edges()


class TestCreateLattice:
    """Tests for lattice graph creation."""

    def test_correct_node_count(self):
        """Lattice should have side^2 nodes."""
        G = create_lattice(5)
        assert G.number_of_nodes() == 25

    def test_lattice_structure(self):
        """Lattice should have correct edge structure."""
        G = create_lattice(3)
        # 3x3 lattice has 12 edges (2*3*2 = 12)
        assert G.number_of_edges() == 12

    def test_positions_in_unit_square(self):
        """Positions should be normalized to [0,1]."""
        G = create_lattice(10)
        for i in range(100):
            pos = G.nodes[i]['pos']
            assert 0 <= pos[0] <= 1
            assert 0 <= pos[1] <= 1


class TestGetPositions:
    """Tests for position extraction."""

    def test_returns_correct_shape(self):
        """Should return N x 2 array."""
        G = create_rgg(30, 0.3, seed=42)
        pos = get_positions(G)
        assert pos.shape == (30, 2)

    def test_positions_match_graph(self):
        """Extracted positions should match graph attributes."""
        G = create_rgg(10, 0.3, seed=42)
        pos = get_positions(G)
        for i in range(10):
            np.testing.assert_array_equal(pos[i], G.nodes[i]['pos'])


class TestGetAdjacency:
    """Tests for adjacency matrix extraction."""

    def test_returns_correct_shape(self):
        """Should return N x N matrix."""
        G = create_lattice(5)
        adj = get_adjacency(G)
        assert adj.shape == (25, 25)

    def test_symmetric(self):
        """Adjacency matrix should be symmetric."""
        G = create_rgg(20, 0.3, seed=42)
        adj = get_adjacency(G)
        np.testing.assert_array_equal(adj, adj.T)

    def test_binary_entries(self):
        """Entries should be 0 or 1."""
        G = create_lattice(5)
        adj = get_adjacency(G)
        assert set(np.unique(adj)).issubset({0, 1})


class TestGetGraphDistances:
    """Tests for shortest path distances."""

    def test_returns_correct_shape(self):
        """Should return N x N matrix."""
        G = create_lattice(4)
        D = get_graph_distances(G)
        assert D.shape == (16, 16)

    def test_diagonal_is_zero(self):
        """Distance to self should be 0."""
        G = create_lattice(4)
        D = get_graph_distances(G)
        np.testing.assert_array_equal(np.diag(D), 0)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        G = create_rgg(20, 0.3, seed=42)
        G = get_largest_component(G)
        D = get_graph_distances(G)
        np.testing.assert_array_equal(D, D.T)

    def test_lattice_known_distances(self):
        """Known lattice distances should be correct."""
        G = create_lattice(3)  # 3x3 lattice
        D = get_graph_distances(G)
        # Corner to corner should be 4 (Manhattan distance)
        # Node 0 is (0,0), node 8 is (2,2)
        assert D[0, 8] == 4


class TestEstimateRadius:
    """Tests for radius estimation."""

    def test_larger_n_smaller_radius(self):
        """More nodes should need smaller radius for same degree."""
        r_small = estimate_radius(50)
        r_large = estimate_radius(200)
        assert r_large < r_small

    def test_achieves_target_degree(self):
        """Estimated radius should achieve approximately target degree."""
        n = 100
        target = 6.0
        radius = estimate_radius(n, target)
        G = create_rgg(n, radius, seed=42)
        actual_degree = 2 * G.number_of_edges() / n
        # Should be within factor of 2
        assert target / 2 < actual_degree < target * 2
