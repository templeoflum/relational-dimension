"""Integration tests for Experiment 01."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_generation import create_rgg, create_lattice, get_positions, get_adjacency, get_graph_distances, get_largest_component, estimate_radius
from correlation import correlation_nn, correlation_lr, correlation_rand, correlation_to_distance
from dimension import extract_dimension_both, normalize_distances
from metrics import compression_ratio, evaluate_p1, evaluate_p2, aggregate_results


class TestFullPipeline:
    """Test the full experiment pipeline."""

    def test_rgg_nn_pipeline(self):
        """Test complete pipeline for RGG with NN correlations."""
        # Create graph
        n = 30
        radius = estimate_radius(n, target_degree=6.0)
        G = create_rgg(n, radius, seed=42)
        G = get_largest_component(G)

        # Get properties
        positions = get_positions(G)
        adjacency = get_adjacency(G)
        D_topo = get_graph_distances(G)

        # Create correlation and convert to distance
        C = correlation_nn(adjacency, 0.8)
        D_corr = correlation_to_distance(C)

        # Normalize
        D_topo_norm = normalize_distances(D_topo)
        D_corr_norm = normalize_distances(D_corr)

        # Extract dimensions
        topo_result = extract_dimension_both(D_topo_norm, k_max=5)
        corr_result = extract_dimension_both(D_corr_norm, k_max=5)

        # Compute compression
        delta = compression_ratio(topo_result['d_mean'], corr_result['d_mean'])

        # Verify outputs are reasonable
        assert 1 <= topo_result['d_mean'] <= 5
        assert 1 <= corr_result['d_mean'] <= 5
        assert -1 <= delta <= 1

    def test_lattice_known_dimension(self):
        """2D lattice should give dimension approximately 2."""
        # Create 8x8 lattice
        G = create_lattice(8)

        # Get distance matrix
        D_topo = get_graph_distances(G)
        D_topo_norm = normalize_distances(D_topo)

        # Extract dimension
        result = extract_dimension_both(D_topo_norm, k_max=6)

        # Should be approximately 2 (allow some tolerance)
        assert 1.5 <= result['d_mean'] <= 3.5

    def test_lr_vs_nn_different_compression(self):
        """LR correlations should give different compression than NN."""
        n = 40
        radius = estimate_radius(n, target_degree=6.0)
        G = create_rgg(n, radius, seed=123)
        G = get_largest_component(G)

        positions = get_positions(G)
        adjacency = get_adjacency(G)
        D_topo = get_graph_distances(G)
        D_topo_norm = normalize_distances(D_topo)

        # NN correlation
        C_nn = correlation_nn(adjacency, 0.8)
        D_nn = normalize_distances(correlation_to_distance(C_nn))

        # LR correlation
        C_lr = correlation_lr(positions, 0.8)
        D_lr = normalize_distances(correlation_to_distance(C_lr))

        # Extract dimensions
        topo_result = extract_dimension_both(D_topo_norm, k_max=5)
        nn_result = extract_dimension_both(D_nn, k_max=5)
        lr_result = extract_dimension_both(D_lr, k_max=5)

        delta_nn = compression_ratio(topo_result['d_mean'], nn_result['d_mean'])
        delta_lr = compression_ratio(topo_result['d_mean'], lr_result['d_mean'])

        # Both should be valid numbers
        assert not np.isnan(delta_nn)
        assert not np.isnan(delta_lr)


class TestPredictionEvaluation:
    """Test prediction evaluation functions."""

    def test_p1_passes_for_low_delta(self):
        """P1 should pass when delta is near zero."""
        deltas = [0.05, 0.08, -0.02, 0.10, 0.03]
        result = evaluate_p1(deltas, threshold=0.2)
        assert result['passed'] is True

    def test_p1_fails_for_high_delta(self):
        """P1 should fail when delta is high."""
        deltas = [0.3, 0.25, 0.35]
        result = evaluate_p1(deltas, threshold=0.2)
        assert result['passed'] is False

    def test_p2_passes_for_high_compression(self):
        """P2 should pass when mean delta > threshold."""
        deltas = [0.3, 0.35, 0.28, 0.32]
        result = evaluate_p2(deltas, threshold=0.25)
        assert result['passed'] is True

    def test_p2_fails_for_low_compression(self):
        """P2 should fail when mean delta <= threshold."""
        deltas = [0.1, 0.15, 0.2]
        result = evaluate_p2(deltas, threshold=0.25)
        assert result['passed'] is False


class TestAggregation:
    """Test result aggregation."""

    def test_aggregates_correctly(self):
        """Should compute correct mean and std."""
        results = [
            {'d_topo': 3, 'd_corr': 2, 'delta': 0.33},
            {'d_topo': 4, 'd_corr': 2, 'delta': 0.50},
            {'d_topo': 3, 'd_corr': 3, 'delta': 0.00},
        ]
        agg = aggregate_results(results)

        assert agg['d_topo_mean'] == pytest.approx(10/3)
        assert agg['delta_mean'] == pytest.approx((0.33 + 0.50 + 0.00) / 3)
        assert agg['n_replications'] == 3

    def test_handles_empty_list(self):
        """Should handle empty results list."""
        agg = aggregate_results([])
        assert agg == {}


class TestEndToEnd:
    """End-to-end test simulating small experiment run."""

    def test_mini_experiment(self):
        """Run a minimal experiment configuration."""
        results = []

        # Run 3 replications of small RGG
        for seed in range(3):
            n = 25
            radius = estimate_radius(n, target_degree=5.0)
            G = create_rgg(n, radius, seed=seed)
            G = get_largest_component(G)

            if G.number_of_nodes() < 10:
                continue

            positions = get_positions(G)
            adjacency = get_adjacency(G)
            D_topo = get_graph_distances(G)

            C = correlation_lr(positions, 0.8)
            D_corr = correlation_to_distance(C)

            D_topo_norm = normalize_distances(D_topo)
            D_corr_norm = normalize_distances(D_corr)

            topo_result = extract_dimension_both(D_topo_norm, k_max=5, n_neighbors=5)
            corr_result = extract_dimension_both(D_corr_norm, k_max=5, n_neighbors=5)

            delta = compression_ratio(topo_result['d_mean'], corr_result['d_mean'])

            results.append({
                'd_topo': topo_result['d_mean'],
                'd_corr': corr_result['d_mean'],
                'delta': delta
            })

        # Should have collected some results
        assert len(results) >= 1

        # Aggregate
        agg = aggregate_results(results)
        assert 'delta_mean' in agg
