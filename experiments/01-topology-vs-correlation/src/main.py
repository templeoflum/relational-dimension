"""
Main runner for Experiment 01: Graph Topology vs Correlation Structure Dimension.

This script runs the full test matrix, computes all metrics, evaluates predictions,
and generates output files and visualizations.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Local imports
from graph_generation import (
    create_rgg, create_lattice, get_positions, get_adjacency,
    get_graph_distances, get_largest_component, estimate_radius
)
from correlation import (
    correlation_nn, correlation_lr, correlation_rand,
    correlation_to_distance, estimate_k_for_density
)
from dimension import extract_dimension_both, normalize_distances
from metrics import (
    compression_ratio, aggregate_results,
    evaluate_p1, evaluate_p2, evaluate_p3, evaluate_p4, evaluate_p5,
    summarize_predictions
)
from visualization import (
    plot_correlation_heatmap, plot_distance_comparison,
    plot_embedding_2d, plot_error_curves_comparison,
    plot_compression_scatter, plot_dose_response,
    plot_predictions_summary, plot_graph_with_correlations
)


# Experiment parameters
EXPERIMENT_ID = "01-topology-vs-correlation"
N_REPLICATIONS = 10
N_NEIGHBORS = 8
ERROR_THRESHOLD = 0.1
K_MAX = 10

# Test matrix
GRAPH_SIZES = [50, 100, 200]
LATTICE_SIDES = [7, 10, 14]  # ~49, 100, 196 nodes
ALPHA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # For dose-response
RHO_DEFAULT = 0.8


def run_single_configuration(graph_type: str, n: int, corr_type: str,
                             rho: float = RHO_DEFAULT,
                             seed: int = None) -> Dict[str, Any]:
    """
    Run a single experiment configuration.

    Args:
        graph_type: 'rgg' or 'lattice'
        n: Number of nodes (or side length for lattice)
        corr_type: 'nn', 'lr', or 'rand'
        rho: Correlation strength
        seed: Random seed

    Returns:
        Dictionary with dimension estimates and metrics
    """
    result = {
        'graph_type': graph_type,
        'n': n,
        'corr_type': corr_type,
        'rho': rho,
        'seed': seed
    }

    # Create graph
    if graph_type == 'rgg':
        radius = estimate_radius(n, target_degree=6.0)
        G = create_rgg(n, radius, seed=seed)
    else:  # lattice
        G = create_lattice(n)

    # Get largest connected component
    G = get_largest_component(G)
    actual_n = G.number_of_nodes()
    result['actual_n'] = actual_n

    # Get graph properties
    positions = get_positions(G)
    adjacency = get_adjacency(G)
    D_topo = get_graph_distances(G)

    # Handle infinite distances (shouldn't happen after connected component)
    if np.any(np.isinf(D_topo)):
        D_topo = np.where(np.isinf(D_topo), actual_n, D_topo)

    # Create correlation matrix
    if corr_type == 'nn':
        C = correlation_nn(adjacency, rho)
    elif corr_type == 'lr':
        C = correlation_lr(positions, rho)
    elif corr_type == 'rand':
        k = estimate_k_for_density(positions, rho)
        C = correlation_rand(actual_n, k, rho, seed=seed)
    else:
        raise ValueError(f"Unknown correlation type: {corr_type}")

    # Convert correlation to distance
    D_corr = correlation_to_distance(C)

    # Normalize distances
    D_topo_norm = normalize_distances(D_topo)
    D_corr_norm = normalize_distances(D_corr)

    # Extract dimensions
    topo_result = extract_dimension_both(D_topo_norm, K_MAX, N_NEIGHBORS)
    corr_result = extract_dimension_both(D_corr_norm, K_MAX, N_NEIGHBORS)

    # Store results
    result['d_topo_isomap'] = topo_result['d_isomap']
    result['d_topo_mds'] = topo_result['d_mds']
    result['d_topo'] = topo_result['d_mean']
    result['d_topo_valid'] = topo_result['valid']

    result['d_corr_isomap'] = corr_result['d_isomap']
    result['d_corr_mds'] = corr_result['d_mds']
    result['d_corr'] = corr_result['d_mean']
    result['d_corr_valid'] = corr_result['valid']

    # Compute compression ratio
    result['delta'] = compression_ratio(result['d_topo'], result['d_corr'])

    # Store error curves for diagnostics
    result['errors_topo_isomap'] = topo_result['errors_isomap'].tolist()
    result['errors_topo_mds'] = topo_result['errors_mds'].tolist()
    result['errors_corr_isomap'] = corr_result['errors_isomap'].tolist()
    result['errors_corr_mds'] = corr_result['errors_mds'].tolist()

    # Store matrices for visualization (only for first replication)
    if seed == 0:
        result['_C'] = C
        result['_D_topo'] = D_topo
        result['_D_corr'] = D_corr
        result['_positions'] = positions
        result['_adjacency'] = adjacency
        result['_embedding_topo'] = topo_result['embedding_isomap']
        result['_embedding_corr'] = corr_result['embedding_isomap']

    return result


def run_baselines(output_dir: Path) -> Dict[str, Any]:
    """
    Run baseline control experiments.

    Returns:
        Dictionary with baseline results
    """
    baselines = {}

    # Baseline 1: Known geometry (2D lattice should give d ≈ 2)
    print("Running baseline: Known geometry (2D lattice)...")
    G = create_lattice(10)  # 10x10 lattice
    positions = get_positions(G)
    adjacency = get_adjacency(G)
    D_topo = get_graph_distances(G)

    C = correlation_nn(adjacency, 0.8)
    D_corr = correlation_to_distance(C)

    topo_result = extract_dimension_both(normalize_distances(D_topo), K_MAX, N_NEIGHBORS)
    corr_result = extract_dimension_both(normalize_distances(D_corr), K_MAX, N_NEIGHBORS)

    baselines['known_geometry'] = {
        'd_topo': topo_result['d_mean'],
        'd_corr': corr_result['d_mean'],
        'expected_d': 2.0,
        'delta': compression_ratio(topo_result['d_mean'], corr_result['d_mean'])
    }

    # Plot error curves for baseline
    plot_error_curves_comparison(
        topo_result['errors_isomap'], topo_result['errors_mds'],
        topo_result['d_isomap'], topo_result['d_mds'],
        output_dir / 'error_curves_baseline_topo.png',
        title='Baseline: Topological Dimension (2D Lattice)'
    )

    # Baseline 2: Identity (no structure)
    print("Running baseline: Identity (no correlations)...")
    n = 50
    C_identity = np.eye(n)
    D_identity = correlation_to_distance(C_identity)

    # For identity, all off-diagonal distances are sqrt(2)
    identity_result = extract_dimension_both(normalize_distances(D_identity), K_MAX, N_NEIGHBORS)

    baselines['identity'] = {
        'd_corr': identity_result['d_mean'],
        'expected_d': n - 1,  # Maximum dimension
        'description': 'Identity correlation (no structure)'
    }

    return baselines


def run_test_matrix(output_dir: Path) -> Dict[str, Any]:
    """
    Run the full test matrix.

    Returns:
        Dictionary with all results organized by configuration
    """
    results = {
        'rgg_nn': {'delta_by_n': {}, 'delta_all': [], 'd_topo_all': [], 'd_corr_all': []},
        'rgg_lr': {'delta_by_n': {}, 'delta_all': [], 'd_topo_all': [], 'd_corr_all': []},
        'rgg_rand': {'delta_by_n': {}, 'delta_all': [], 'd_topo_all': [], 'd_corr_all': []},
        'lattice_nn': {'delta_by_n': {}, 'delta_all': [], 'd_topo_all': [], 'd_corr_all': []},
        'lattice_lr': {'delta_by_n': {}, 'delta_all': [], 'd_topo_all': [], 'd_corr_all': []},
    }

    # RGG configurations
    configs = [
        ('rgg', 'nn', GRAPH_SIZES),
        ('rgg', 'lr', GRAPH_SIZES),
        ('rgg', 'rand', GRAPH_SIZES),
    ]

    # Lattice configurations
    configs += [
        ('lattice', 'nn', LATTICE_SIDES),
        ('lattice', 'lr', LATTICE_SIDES),
    ]

    total_runs = sum(len(sizes) * N_REPLICATIONS for _, _, sizes in configs)

    with tqdm(total=total_runs, desc="Running test matrix") as pbar:
        for graph_type, corr_type, sizes in configs:
            config_key = f'{graph_type}_{corr_type}'

            for size in sizes:
                size_results = []
                results[config_key]['delta_by_n'][size] = []

                for rep in range(N_REPLICATIONS):
                    seed = rep
                    result = run_single_configuration(
                        graph_type, size, corr_type, RHO_DEFAULT, seed
                    )
                    size_results.append(result)

                    # Store delta
                    results[config_key]['delta_by_n'][size].append(result['delta'])
                    results[config_key]['delta_all'].append(result['delta'])
                    results[config_key]['d_topo_all'].append(result['d_topo'])
                    results[config_key]['d_corr_all'].append(result['d_corr'])

                    # Generate visualizations for first replication
                    if rep == 0 and '_C' in result:
                        prefix = f'{config_key}_n{size}'
                        plot_correlation_heatmap(
                            result['_C'],
                            f'Correlation Matrix: {config_key} N={size}',
                            output_dir / f'correlation_{prefix}.png'
                        )
                        plot_distance_comparison(
                            result['_D_topo'], result['_D_corr'],
                            output_dir / f'distance_comparison_{prefix}.png',
                            title=f'Distance Comparison: {config_key} N={size}'
                        )
                        if result['_embedding_topo'].shape[1] >= 2:
                            plot_embedding_2d(
                                result['_embedding_topo'],
                                output_dir / f'embedding_topo_{prefix}.png',
                                title=f'Topological Embedding: {config_key} N={size}'
                            )
                            plot_embedding_2d(
                                result['_embedding_corr'],
                                output_dir / f'embedding_corr_{prefix}.png',
                                title=f'Correlation Embedding: {config_key} N={size}'
                            )

                    pbar.update(1)

                # Aggregate for this size
                agg = aggregate_results(size_results)
                results[config_key][f'n{size}_aggregated'] = agg

    return results


def run_dose_response(output_dir: Path) -> Dict[str, Any]:
    """
    Run dose-response experiment: vary correlation strength.

    Returns:
        Dictionary with dose-response results
    """
    print("Running dose-response experiment...")
    alpha_values = []
    delta_values_mean = []
    delta_values_std = []

    for alpha in tqdm(ALPHA_VALUES, desc="Dose-response"):
        deltas = []
        for rep in range(N_REPLICATIONS):
            # Use RGG with LR correlations, vary rho
            rho = 0.3 + 0.6 * alpha  # Map alpha [0,1] to rho [0.3, 0.9]

            result = run_single_configuration(
                'rgg', 100, 'lr', rho=rho, seed=rep
            )
            deltas.append(result['delta'])

        alpha_values.append(alpha)
        delta_values_mean.append(np.mean(deltas))
        delta_values_std.append(np.std(deltas))

    return {
        'alpha_values': alpha_values,
        'delta_values': delta_values_mean,
        'delta_std': delta_values_std
    }


def main():
    """Main entry point for experiment."""
    print(f"=" * 60)
    print(f"Experiment 01: Graph Topology vs Correlation Structure Dimension")
    print(f"=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Setup output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / 'output'
    reports_dir = script_dir / 'reports'
    output_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # Run baselines
    print("\n--- Running Baselines ---")
    baselines = run_baselines(output_dir)
    print(f"Known geometry baseline: d_topo={baselines['known_geometry']['d_topo']:.2f}, "
          f"d_corr={baselines['known_geometry']['d_corr']:.2f} (expected ~2.0)")

    # Run test matrix
    print("\n--- Running Test Matrix ---")
    results = run_test_matrix(output_dir)

    # Run dose-response
    print("\n--- Running Dose-Response ---")
    dose_response = run_dose_response(output_dir)
    results['dose_response'] = dose_response

    # Evaluate predictions
    print("\n--- Evaluating Predictions ---")
    predictions = {}

    # P1: Baseline agreement (NN correlations)
    predictions['P1'] = evaluate_p1(results['rgg_nn']['delta_all'])
    print(f"P1 (Baseline): {'PASS' if predictions['P1']['passed'] else 'FAIL'} "
          f"(measured={predictions['P1']['measured']:.3f}, threshold={predictions['P1']['threshold']})")

    # P2: Long-range compression
    predictions['P2'] = evaluate_p2(results['rgg_lr']['delta_all'])
    print(f"P2 (Compression): {'PASS' if predictions['P2']['passed'] else 'FAIL'} "
          f"(measured={predictions['P2']['measured']:.3f}, threshold={predictions['P2']['threshold']})")

    # P3: Scaling behavior
    small_n = min(results['rgg_lr']['delta_by_n'].keys())
    large_n = max(results['rgg_lr']['delta_by_n'].keys())
    predictions['P3'] = evaluate_p3(
        results['rgg_lr']['delta_by_n'][small_n],
        results['rgg_lr']['delta_by_n'][large_n]
    )
    print(f"P3 (Scaling): {'PASS' if predictions['P3']['passed'] else 'FAIL'} "
          f"(ratio={predictions['P3']['measured']:.3f}, threshold={predictions['P3']['threshold']})")

    # P4: Random control
    predictions['P4'] = evaluate_p4(
        results['rgg_rand']['delta_all'],
        results['rgg_lr']['delta_all']
    )
    print(f"P4 (Control): {'PASS' if predictions['P4']['passed'] else 'FAIL'} "
          f"(random={predictions['P4']['measured_random']:.3f}, lr={predictions['P4']['measured_lr']:.3f})")

    # P5: Dose-response
    predictions['P5'] = evaluate_p5(
        dose_response['alpha_values'],
        dose_response['delta_values']
    )
    print(f"P5 (Dose-Response): {'PASS' if predictions['P5']['passed'] else 'FAIL'} "
          f"(r={predictions['P5']['measured']:.3f}, threshold={predictions['P5']['threshold']})")

    # Generate summary
    summary = summarize_predictions(predictions)
    print(f"\n--- Summary ---")
    print(f"Predictions passed: {summary['predictions_passed']}/{summary['predictions_total']}")
    print(f"Status: {summary['status'].upper()}")
    print(f"Core hypothesis (P2): {'SUPPORTED' if summary['core_hypothesis_passed'] else 'FALSIFIED'}")

    # Generate report figures
    print("\n--- Generating Report Figures ---")
    plot_compression_scatter(results, reports_dir / 'compression_scatter.png')
    plot_dose_response(
        dose_response['alpha_values'],
        dose_response['delta_values'],
        reports_dir / 'dose_response.png',
        r_value=predictions['P5']['measured']
    )
    plot_predictions_summary(predictions, reports_dir / 'predictions_summary.png')

    # Prepare output metrics
    # Convert numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Skip internal keys (those starting with _)
                if isinstance(k, str) and k.startswith('_'):
                    continue
                # Convert integer keys to strings for JSON
                key = str(k) if isinstance(k, (int, np.integer)) else k
                result[key] = convert_for_json(v)
            return result
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj

    metrics = {
        'experiment': EXPERIMENT_ID,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_replications': N_REPLICATIONS,
            'n_neighbors': N_NEIGHBORS,
            'error_threshold': ERROR_THRESHOLD,
            'k_max': K_MAX,
            'graph_sizes': GRAPH_SIZES,
            'lattice_sides': LATTICE_SIDES,
            'rho_default': RHO_DEFAULT,
            'alpha_values': ALPHA_VALUES
        },
        'baselines': convert_for_json(baselines),
        'results': convert_for_json(results),
        'predictions': convert_for_json(predictions),
        'summary': summary
    }

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Print completion
    print(f"\n{'=' * 60}")
    print(f"Experiment complete: {datetime.now().isoformat()}")
    print(f"Output directory: {output_dir}")
    print(f"Reports directory: {reports_dir}")
    print(f"{'=' * 60}")

    return metrics


if __name__ == '__main__':
    main()
