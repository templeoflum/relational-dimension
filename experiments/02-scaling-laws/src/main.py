"""
Main runner for Experiment 02: Compression Scaling Laws.

This script runs the scaling test matrix, fits log/power models,
evaluates predictions P1-P6, and generates output files and visualizations.
"""

import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Add parent experiment to path for imports
exp01_path = Path(__file__).parent.parent.parent / '01-topology-vs-correlation' / 'src'
sys.path.insert(0, str(exp01_path))

# Imports from Experiment 01
from graph_generation import (
    create_rgg, get_positions, get_adjacency,
    get_graph_distances, get_largest_component, estimate_radius
)
from correlation import correlation_lr, correlation_to_distance
from dimension import normalize_distances
from metrics import compression_ratio

# Local imports
from dimension_improved import continuous_dimension_both
from scaling_analysis import (
    fit_log_model, fit_power_model, predict_delta, compare_models,
    evaluate_all_predictions, summarize_predictions
)

# Experiment parameters
EXPERIMENT_ID = "02-scaling-laws"
RHO_DEFAULT = 0.8
K_MAX = 10
N_NEIGHBORS = 8

# Test matrix
TEST_MATRIX = [
    (50, 10),    # N, replications
    (100, 10),
    (200, 10),
    (300, 10),
    (500, 10),
    (750, 5),
    (1000, 5),
]


def run_single_configuration(n: int, seed: int = None,
                             use_sparse: bool = False) -> Dict[str, Any]:
    """
    Run a single experiment configuration with improved dimension estimation.

    Args:
        n: Number of nodes
        seed: Random seed
        use_sparse: Whether to use sparse methods for large N

    Returns:
        Dictionary with dimension estimates and metrics
    """
    result = {
        'n': n,
        'seed': seed,
        'use_sparse': use_sparse
    }

    # Create RGG
    radius = estimate_radius(n, target_degree=6.0)
    G = create_rgg(n, radius, seed=seed)

    # Get largest connected component
    G = get_largest_component(G)
    actual_n = G.number_of_nodes()
    result['actual_n'] = actual_n

    # Get graph properties
    positions = get_positions(G)
    D_topo = get_graph_distances(G)

    # Handle infinite distances
    if np.any(np.isinf(D_topo)):
        D_topo = np.where(np.isinf(D_topo), actual_n, D_topo)

    # Create LR correlation matrix
    C = correlation_lr(positions, RHO_DEFAULT)
    D_corr = correlation_to_distance(C)

    # Normalize distances
    D_topo_norm = normalize_distances(D_topo)
    D_corr_norm = normalize_distances(D_corr)

    # Extract dimensions with improved method
    topo_result = continuous_dimension_both(D_topo_norm, K_MAX, N_NEIGHBORS, use_sparse)
    corr_result = continuous_dimension_both(D_corr_norm, K_MAX, N_NEIGHBORS, use_sparse)

    # Store results
    result['d_topo_isomap'] = float(topo_result['d_isomap'])
    result['d_topo_mds'] = float(topo_result['d_mds'])
    result['d_topo'] = float(topo_result['d_mean'])

    result['d_corr_isomap'] = float(corr_result['d_isomap'])
    result['d_corr_mds'] = float(corr_result['d_mds'])
    result['d_corr'] = float(corr_result['d_mean'])

    # Compute compression ratio
    result['delta'] = compression_ratio(result['d_topo'], result['d_corr'])

    # Store variance curves for diagnostics
    result['variances_topo_isomap'] = topo_result['variances_isomap'].tolist()
    result['variances_corr_isomap'] = corr_result['variances_isomap'].tolist()

    return result


def run_test_matrix() -> Dict[str, Any]:
    """
    Run the full test matrix for scaling analysis.

    Returns:
        Dictionary with all results organized by N
    """
    results = {
        'delta_by_n': {},
        'std_by_n': {},
        'raw_results': [],
        'N_values': [],
        'delta_means': [],
        'delta_stds': []
    }

    total_runs = sum(reps for _, reps in TEST_MATRIX)

    with tqdm(total=total_runs, desc="Running scaling test matrix") as pbar:
        for n, replications in TEST_MATRIX:
            # Use sparse methods for large N
            use_sparse = n > 500

            deltas = []
            n_results = []

            for rep in range(replications):
                seed = rep
                result = run_single_configuration(n, seed, use_sparse)
                n_results.append(result)
                deltas.append(result['delta'])
                results['raw_results'].append(result)
                pbar.update(1)

            # Store aggregated results
            results['delta_by_n'][n] = deltas
            results['std_by_n'][n] = float(np.std(deltas))
            results['N_values'].append(n)
            results['delta_means'].append(float(np.mean(deltas)))
            results['delta_stds'].append(float(np.std(deltas)))

    return results


def fit_scaling_models(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fit log and power models to scaling data.

    Args:
        results: Results from test matrix

    Returns:
        Updated results with fitted models
    """
    N_values = np.array(results['N_values'])
    delta_means = np.array(results['delta_means'])

    # Fit on N <= 500 for predictive validity test
    train_mask = N_values <= 500
    N_train = N_values[train_mask]
    delta_train = delta_means[train_mask]

    # Fit both models on training data
    log_model = fit_log_model(N_train, delta_train)
    power_model = fit_power_model(N_train, delta_train)

    results['log_model'] = log_model
    results['power_model'] = power_model

    # Compare models
    results['model_comparison'] = compare_models(log_model, power_model)

    # Predict delta at N=1000
    best_model = log_model if results['model_comparison']['winner'] == 'log' else power_model
    results['delta_predicted_1000'] = predict_delta(best_model, 1000)
    results['delta_measured_1000'] = float(np.mean(results['delta_by_n'].get(1000, [0])))

    # Also fit full models (on all data) for visualization
    log_model_full = fit_log_model(N_values, delta_means)
    power_model_full = fit_power_model(N_values, delta_means)
    results['log_model_full'] = log_model_full
    results['power_model_full'] = power_model_full

    return results


def generate_visualizations(results: Dict[str, Any], output_dir: Path,
                            reports_dir: Path) -> None:
    """
    Generate visualization figures.

    Args:
        results: Full results dictionary
        output_dir: Directory for output files
        reports_dir: Directory for report figures
    """
    import matplotlib.pyplot as plt

    # Setup style
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    N_values = np.array(results['N_values'])
    delta_means = np.array(results['delta_means'])
    delta_stds = np.array(results['delta_stds'])

    # 1. Scaling curve with fitted models
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data points with error bars
    ax.errorbar(N_values, delta_means, yerr=delta_stds, fmt='bo-',
                capsize=4, markersize=8, linewidth=2, label='Measured')

    # Log model fit
    if results['log_model_full'].get('valid', False):
        N_smooth = np.linspace(min(N_values), max(N_values), 100)
        delta_log = results['log_model_full']['a'] * np.log(N_smooth) + results['log_model_full']['b']
        ax.plot(N_smooth, delta_log, 'g--', linewidth=2,
                label=f"Log: {results['log_model_full']['formula']} (R²={results['log_model_full']['r2']:.3f})")

    # Power model fit
    if results['power_model_full'].get('valid', False):
        N_smooth = np.linspace(min(N_values), max(N_values), 100)
        delta_power = results['power_model_full']['c'] * (N_smooth ** results['power_model_full']['alpha'])
        ax.plot(N_smooth, delta_power, 'r:', linewidth=2,
                label=f"Power: {results['power_model_full']['formula']} (R²={results['power_model_full']['r2']:.3f})")

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Graph Size (N)')
    ax.set_ylabel('Compression Ratio (δ)')
    ax.set_title('Compression Scaling with Graph Size')
    ax.legend()
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(reports_dir / 'scaling_curve.png')
    plt.close()

    # 2. Residuals plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if results['log_model_full'].get('valid', False):
        residuals_log = results['log_model_full']['residuals']
        ax1.scatter(N_values, residuals_log, s=60, alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Graph Size (N)')
        ax1.set_ylabel('Residual')
        ax1.set_title('Log Model Residuals')
        ax1.set_xscale('log')

    if results['power_model_full'].get('valid', False):
        residuals_power = results['power_model_full']['residuals']
        ax2.scatter(N_values, residuals_power, s=60, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Graph Size (N)')
        ax2.set_ylabel('Residual')
        ax2.set_title('Power Model Residuals')
        ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(reports_dir / 'residuals.png')
    plt.close()

    # 3. Prediction test visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Training data (N <= 500)
    train_mask = N_values <= 500
    ax.errorbar(N_values[train_mask], delta_means[train_mask],
                yerr=delta_stds[train_mask], fmt='bo-',
                capsize=4, markersize=8, linewidth=2, label='Training data (N ≤ 500)')

    # Test data (N > 500)
    test_mask = N_values > 500
    ax.errorbar(N_values[test_mask], delta_means[test_mask],
                yerr=delta_stds[test_mask], fmt='rs-',
                capsize=4, markersize=10, linewidth=2, label='Test data (N > 500)')

    # Predicted value at N=1000
    delta_pred_1000 = results['delta_predicted_1000']
    ax.scatter([1000], [delta_pred_1000], c='green', s=150, marker='^',
               zorder=5, label=f'Predicted δ(1000) = {delta_pred_1000:.3f}')

    # Extrapolation line
    if results['log_model'].get('valid', False):
        N_smooth = np.linspace(min(N_values), 1100, 100)
        delta_pred = results['log_model']['a'] * np.log(N_smooth) + results['log_model']['b']
        ax.plot(N_smooth, delta_pred, 'g--', alpha=0.5, label='Model extrapolation')

    ax.set_xlabel('Graph Size (N)')
    ax.set_ylabel('Compression Ratio (δ)')
    ax.set_title('Predictive Validity Test: N=1000')
    ax.legend()
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(reports_dir / 'prediction_test.png')
    plt.close()

    # 4. Predictions summary
    predictions = results.get('predictions', {})
    if predictions:
        fig, ax = plt.subplots(figsize=(12, 6))

        names = list(predictions.keys())
        passed = [1 if predictions[p].get('passed', False) else 0 for p in names]
        colors = ['green' if p else 'red' for p in passed]

        bars = ax.bar(names, [1] * len(names), color=colors, alpha=0.7, edgecolor='black')

        for i, (name, pred) in enumerate(predictions.items()):
            measured = pred.get('measured', 'N/A')
            threshold = pred.get('threshold', 'N/A')
            if isinstance(measured, float):
                measured = f'{measured:.3f}'
            if isinstance(threshold, float):
                threshold = f'{threshold:.3f}'
            ax.annotate(f'Measured: {measured}\nThreshold: {threshold}',
                        xy=(i, 0.5), ha='center', va='center', fontsize=9)

        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Pass (1) / Fail (0)')
        ax.set_title('Prediction Results (P1-P6)')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['FAIL', 'PASS'])

        plt.tight_layout()
        plt.savefig(reports_dir / 'predictions_summary.png')
        plt.close()


def main():
    """Main entry point for experiment."""
    print("=" * 60)
    print("Experiment 02: Compression Scaling Laws")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Setup output directories
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / 'output'
    reports_dir = script_dir / 'reports'
    output_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # Run test matrix
    print("\n--- Running Test Matrix ---")
    results = run_test_matrix()

    # Print intermediate results
    print("\n--- Scaling Data ---")
    for n, mean, std in zip(results['N_values'], results['delta_means'], results['delta_stds']):
        print(f"  N={n:4d}: delta = {mean:.4f} +/- {std:.4f}")

    # Fit scaling models
    print("\n--- Fitting Scaling Models ---")
    results = fit_scaling_models(results)

    print(f"  Log model:   {results['log_model'].get('formula', 'N/A')} (R^2 = {results['log_model'].get('r2', 0):.4f})")
    print(f"  Power model: {results['power_model'].get('formula', 'N/A')} (R^2 = {results['power_model'].get('r2', 0):.4f})")
    print(f"  Winner: {results['model_comparison']['winner']}")

    # Evaluate predictions
    print("\n--- Evaluating Predictions ---")
    predictions = evaluate_all_predictions(results)
    results['predictions'] = predictions

    for name, pred in predictions.items():
        status = 'PASS' if pred.get('passed', False) else 'FAIL'
        measured = pred.get('measured', 'N/A')
        if isinstance(measured, float):
            measured = f'{measured:.4f}'
        print(f"  {name} ({pred.get('description', '')}): {status} "
              f"(measured={measured}, threshold={pred.get('threshold', 'N/A')})")

    # Summary
    summary = summarize_predictions(predictions)
    results['summary'] = summary

    print(f"\n--- Summary ---")
    print(f"Predictions passed: {summary['predictions_passed']}/{summary['predictions_total']}")
    print(f"Status: {summary['status'].upper()}")
    print(f"Monotonic scaling: {'YES' if summary['monotonic_scaling_passed'] else 'NO'}")
    print(f"Functional form: {summary['functional_form']}")
    print(f"Predictive validity: {'YES' if summary['predictive_validity'] else 'NO'}")

    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    generate_visualizations(results, output_dir, reports_dir)

    # Convert for JSON serialization
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
                if isinstance(k, str) and k.startswith('_'):
                    continue
                key = str(k) if isinstance(k, (int, np.integer)) else k
                result[key] = convert_for_json(v)
            return result
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj

    # Prepare output metrics
    metrics = {
        'experiment': EXPERIMENT_ID,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'rho_default': RHO_DEFAULT,
            'k_max': K_MAX,
            'n_neighbors': N_NEIGHBORS,
            'test_matrix': TEST_MATRIX
        },
        'scaling_data': {
            'N_values': results['N_values'],
            'delta_means': results['delta_means'],
            'delta_stds': results['delta_stds'],
            'delta_by_n': convert_for_json(results['delta_by_n']),
        },
        'models': {
            'log_model': convert_for_json(results['log_model']),
            'power_model': convert_for_json(results['power_model']),
            'comparison': results['model_comparison'],
            'log_model_full': convert_for_json(results['log_model_full']),
            'power_model_full': convert_for_json(results['power_model_full']),
        },
        'prediction_test': {
            'delta_predicted_1000': results['delta_predicted_1000'],
            'delta_measured_1000': results['delta_measured_1000'],
        },
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
