"""
Main experiment runner for Experiment 03: Methodological Consistency.

Runs the test matrix with consistent full methods across all graph sizes.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent experiments to path for module reuse
exp_dir = Path(__file__).parent.parent
project_root = exp_dir.parent.parent
sys.path.insert(0, str(project_root / 'experiments' / '01-topology-vs-correlation' / 'src'))

from graph_generation import create_rgg, get_positions, get_graph_distances, estimate_radius
from correlation import correlation_lr, correlation_to_distance
from consistent_methods import extract_dimension_consistent, compute_compression_ratio
from scaling_robust import (fit_log_model, fit_sqrt_model, compare_models,
                            predict_delta, evaluate_all_predictions, summarize_predictions)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Test matrix configuration
TEST_MATRIX = [
    (50, 20),    # N, replications
    (100, 20),
    (200, 20),
    (300, 15),
    (500, 15),
    (750, 15),
    (1000, 15),
]

# Parameters
RHO_DEFAULT = 0.8
K_MAX = 10
N_NEIGHBORS = 8
TRAINING_N_MAX = 500  # Fit models on N <= this value


def run_single_configuration(n: int, seed: int, rho: float = RHO_DEFAULT) -> dict:
    """
    Run a single graph configuration.

    Returns dimension estimates and compression ratio.
    """
    # Generate graph
    radius = estimate_radius(n, target_degree=6.0)
    G = create_rgg(n, radius, seed=seed)
    positions = get_positions(G)

    # Get topological distances
    D_topo = get_graph_distances(G)
    n_actual = D_topo.shape[0]  # May be smaller if graph disconnected

    if n_actual < 10:
        return {'valid': False, 'error': 'Graph too small after component extraction'}

    # Generate long-range correlations
    positions_subset = positions[:n_actual]
    C = correlation_lr(positions_subset, rho=rho)
    D_corr = correlation_to_distance(C)

    # Extract dimensions using consistent methods
    topo_result = extract_dimension_consistent(D_topo, K_MAX, N_NEIGHBORS)
    corr_result = extract_dimension_consistent(D_corr, K_MAX, N_NEIGHBORS)

    # Get fractional dimensions for smoother estimates
    d_topo = topo_result['d_fractional_mean']
    d_corr = corr_result['d_fractional_mean']

    # Compute compression ratio
    delta = compute_compression_ratio(d_topo, d_corr)

    return {
        'n': n,
        'n_actual': n_actual,
        'seed': seed,
        'd_topo': d_topo,
        'd_corr': d_corr,
        'delta': delta,
        'd_topo_mds': topo_result['d_mds'],
        'd_topo_isomap': topo_result['d_isomap'],
        'd_corr_mds': corr_result['d_mds'],
        'd_corr_isomap': corr_result['d_isomap'],
        'topo_valid': topo_result['valid'],
        'corr_valid': corr_result['valid'],
        'topo_agreement': topo_result['agreement_diff'],
        'corr_agreement': corr_result['agreement_diff'],
        'stress': topo_result['stress'],
        'valid': True
    }


def run_test_matrix(test_matrix: list, verbose: bool = True) -> dict:
    """
    Run the full test matrix.
    """
    results_by_n = {}
    all_results = []

    total_configs = sum(reps for _, reps in test_matrix)
    config_count = 0

    for n, n_reps in test_matrix:
        if verbose:
            print(f"\nRunning N={n} ({n_reps} replications)...")

        results_by_n[n] = []

        for rep in range(n_reps):
            seed = n * 1000 + rep
            result = run_single_configuration(n, seed)

            if result.get('valid', False):
                results_by_n[n].append(result)
                all_results.append(result)

            config_count += 1
            if verbose and (config_count % 10 == 0 or config_count == total_configs):
                print(f"  Progress: {config_count}/{total_configs}")

    return {
        'results_by_n': results_by_n,
        'all_results': all_results
    }


def compute_scaling_statistics(results_by_n: dict) -> dict:
    """
    Compute scaling statistics from results.
    """
    N_values = []
    delta_means = []
    delta_stds = []
    delta_ses = []
    delta_by_n = {}
    se_by_n = {}

    for n in sorted(results_by_n.keys()):
        results = results_by_n[n]
        if not results:
            continue

        deltas = [r['delta'] for r in results]
        N_values.append(n)
        delta_means.append(np.mean(deltas))
        delta_stds.append(np.std(deltas))
        delta_ses.append(np.std(deltas) / np.sqrt(len(deltas)))
        delta_by_n[n] = deltas
        se_by_n[n] = np.std(deltas) / np.sqrt(len(deltas))

    return {
        'N_values': N_values,
        'delta_means': delta_means,
        'delta_stds': delta_stds,
        'delta_ses': delta_ses,
        'delta_by_n': delta_by_n,
        'se_by_n': se_by_n
    }


def compute_validation_metrics(all_results: list) -> dict:
    """
    Compute validation metrics for quality checks.
    """
    if not all_results:
        return {}

    # Method agreement rate
    topo_agreements = [r['topo_agreement'] for r in all_results if 'topo_agreement' in r]
    corr_agreements = [r['corr_agreement'] for r in all_results if 'corr_agreement' in r]

    agreement_ok = sum(1 for a in topo_agreements + corr_agreements if a <= 0.5)
    total_checks = len(topo_agreements) + len(corr_agreements)
    agreement_rate = agreement_ok / total_checks if total_checks > 0 else 0

    # Mean stress
    stresses = [r['stress'] for r in all_results if 'stress' in r]
    mean_stress = np.mean(stresses) if stresses else 0

    # Continuity check (no sign reversal between N=500 and N=750)
    deltas_500 = [r['delta'] for r in all_results if r.get('n') == 500]
    deltas_750 = [r['delta'] for r in all_results if r.get('n') == 750]

    if deltas_500 and deltas_750:
        mean_500 = np.mean(deltas_500)
        mean_750 = np.mean(deltas_750)
        # Check if both have same sign or gap is small
        continuity_preserved = (mean_500 * mean_750 > 0) or abs(mean_750 - mean_500) < 0.15
    else:
        continuity_preserved = True

    return {
        'method_agreement_rate': float(agreement_rate),
        'mean_stress': float(mean_stress),
        'continuity_preserved': bool(continuity_preserved),
        'n_valid_samples': len(all_results)
    }


def fit_scaling_models(scaling_stats: dict) -> dict:
    """
    Fit log and sqrt models on training set (N <= 500).
    """
    N_values = np.array(scaling_stats['N_values'])
    delta_means = np.array(scaling_stats['delta_means'])

    # Training set: N <= 500
    train_mask = N_values <= TRAINING_N_MAX
    N_train = N_values[train_mask]
    delta_train = delta_means[train_mask]

    log_model = fit_log_model(N_train, delta_train)
    sqrt_model = fit_sqrt_model(N_train, delta_train)
    comparison = compare_models(log_model, sqrt_model)

    # Predict at N=1000
    delta_pred_1000 = predict_delta(log_model, 1000)
    delta_meas_1000 = scaling_stats['delta_by_n'].get(1000, [0])
    delta_meas_1000 = np.mean(delta_meas_1000) if delta_meas_1000 else 0

    return {
        'log_model': log_model,
        'sqrt_model': sqrt_model,
        'comparison': comparison,
        'prediction_test': {
            'delta_predicted_1000': delta_pred_1000,
            'delta_measured_1000': delta_meas_1000
        }
    }


def create_visualizations(scaling_stats: dict, models: dict, predictions: dict,
                          output_dir: Path):
    """
    Create all visualization figures.
    """
    reports_dir = output_dir.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)

    N_values = np.array(scaling_stats['N_values'])
    delta_means = np.array(scaling_stats['delta_means'])
    delta_stds = np.array(scaling_stats['delta_stds'])
    delta_ses = np.array(scaling_stats['delta_ses'])

    # 1. Scaling curve with fitted model
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(N_values, delta_means, yerr=delta_ses * 1.96, fmt='o-',
                capsize=5, capthick=2, markersize=8, label='Measured (95% CI)')

    # Plot fitted log model
    log_model = models['log_model']
    if log_model.get('valid', False):
        N_fit = np.linspace(50, 1000, 100)
        delta_fit = log_model['a'] * np.log(N_fit) + log_model['b']
        ax.plot(N_fit, delta_fit, 'r--', linewidth=2,
                label=f"Log model: R^2={log_model['r2']:.3f}")

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=500, color='green', linestyle='--', alpha=0.5, label='Training/Test boundary')

    ax.set_xlabel('Graph Size N', fontsize=12)
    ax.set_ylabel('Compression Ratio delta', fontsize=12)
    ax.set_title('Compression Scaling with Consistent Methods', fontsize=14)
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(reports_dir / 'scaling_curve.png', dpi=150)
    plt.close()

    # 2. Method comparison (MDS vs Isomap agreement)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect agreement data
    agreement_data = {}
    for n, deltas in scaling_stats['delta_by_n'].items():
        if n not in agreement_data:
            agreement_data[n] = len(deltas)

    ax.bar(range(len(agreement_data)), list(agreement_data.values()),
           tick_label=[str(n) for n in agreement_data.keys()])
    ax.set_xlabel('Graph Size N', fontsize=12)
    ax.set_ylabel('Valid Samples', fontsize=12)
    ax.set_title('Valid Samples per Graph Size', fontsize=14)

    plt.tight_layout()
    plt.savefig(reports_dir / 'method_comparison.png', dpi=150)
    plt.close()

    # 3. Residuals plot
    fig, ax = plt.subplots(figsize=(10, 6))

    log_model = models['log_model']
    if log_model.get('valid', False):
        train_mask = N_values <= TRAINING_N_MAX
        N_train = N_values[train_mask]
        residuals = log_model.get('residuals', [])

        if len(residuals) == len(N_train):
            ax.scatter(N_train, residuals, s=100, alpha=0.7)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_xlabel('Graph Size N', fontsize=12)
            ax.set_ylabel('Residual (Measured - Predicted)', fontsize=12)
            ax.set_title('Log Model Residuals (Training Set)', fontsize=14)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(reports_dir / 'residuals.png', dpi=150)
    plt.close()

    # 4. Predictions summary
    fig, ax = plt.subplots(figsize=(12, 6))

    pred_ids = []
    pred_passed = []
    pred_values = []
    pred_thresholds = []

    for pid in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        if pid in predictions:
            p = predictions[pid]
            pred_ids.append(pid)
            pred_passed.append(p.get('passed', False))
            pred_values.append(p.get('measured', 0))
            pred_thresholds.append(p.get('threshold', 0))

    colors = ['green' if p else 'red' for p in pred_passed]
    bars = ax.bar(pred_ids, pred_values, color=colors, alpha=0.7, edgecolor='black')

    # Add threshold lines
    for i, (pid, thresh) in enumerate(zip(pred_ids, pred_thresholds)):
        if pid in ['P1', 'P2']:  # Greater than thresholds
            ax.hlines(thresh, i - 0.4, i + 0.4, colors='blue', linestyles='--', linewidth=2)
        elif pid in ['P3', 'P4', 'P6']:  # Less than thresholds
            ax.hlines(thresh, i - 0.4, i + 0.4, colors='blue', linestyles='--', linewidth=2)

    ax.axhline(y=0, color='gray', linestyle=':')
    ax.set_xlabel('Prediction', fontsize=12)
    ax.set_ylabel('Measured Value', fontsize=12)
    ax.set_title('Prediction Results (Green=Pass, Red=Fail, Blue=Threshold)', fontsize=14)

    # Add pass/fail labels
    for i, (passed, val) in enumerate(zip(pred_passed, pred_values)):
        label = 'PASS' if passed else 'FAIL'
        y_pos = val + 0.02 if val >= 0 else val - 0.05
        ax.text(i, y_pos, label, ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(reports_dir / 'predictions_summary.png', dpi=150)
    plt.close()

    print(f"Visualizations saved to {reports_dir}")


def main():
    """
    Main experiment runner.
    """
    print("=" * 60)
    print("Experiment 03: Methodological Consistency and Robust Scaling")
    print("=" * 60)

    # Setup paths
    exp_dir = Path(__file__).parent.parent
    output_dir = exp_dir / 'output'
    output_dir.mkdir(exist_ok=True)

    # Run test matrix
    print("\nRunning test matrix...")
    raw_results = run_test_matrix(TEST_MATRIX, verbose=True)

    # Compute statistics
    print("\nComputing scaling statistics...")
    scaling_stats = compute_scaling_statistics(raw_results['results_by_n'])

    # Compute validation metrics
    validation = compute_validation_metrics(raw_results['all_results'])

    # Fit models
    print("\nFitting scaling models...")
    models = fit_scaling_models(scaling_stats)

    # Prepare full results for prediction evaluation
    full_results = {
        **scaling_stats,
        **models
    }

    # Evaluate predictions
    print("\nEvaluating predictions...")
    predictions = evaluate_all_predictions(full_results)
    summary = summarize_predictions(predictions)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nScaling Data:")
    print(f"{'N':>6} {'delta (mean)':>12} {'delta (std)':>12} {'SE':>8} {'n_reps':>8}")
    print("-" * 50)
    for i, n in enumerate(scaling_stats['N_values']):
        print(f"{n:>6} {scaling_stats['delta_means'][i]:>12.4f} "
              f"{scaling_stats['delta_stds'][i]:>12.4f} "
              f"{scaling_stats['delta_ses'][i]:>8.4f} "
              f"{len(scaling_stats['delta_by_n'][n]):>8}")

    print(f"\nLog Model: {models['log_model'].get('formula', 'N/A')}")
    print(f"  R^2 = {models['log_model'].get('r2', 0):.4f}")

    print(f"\nSqrt Model: {models['sqrt_model'].get('formula', 'N/A')}")
    print(f"  R^2 = {models['sqrt_model'].get('r2', 0):.4f}")

    print(f"\nBest Model: {models['comparison'].get('winner', 'N/A')}")

    print("\n" + "-" * 60)
    print("PREDICTION RESULTS")
    print("-" * 60)

    for pid in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        if pid in predictions:
            p = predictions[pid]
            status = 'PASS' if p['passed'] else 'FAIL'
            print(f"{pid}: {p['description']:25} | {status:4} | "
                  f"measured={p['measured']:.4f} threshold={p['threshold']}")

    print(f"\nPredictions Passed: {summary['predictions_passed']}/{summary['predictions_total']}")
    print(f"Status: {summary['status']}")

    # Validation metrics
    print("\n" + "-" * 60)
    print("VALIDATION METRICS")
    print("-" * 60)
    print(f"Method Agreement Rate: {validation.get('method_agreement_rate', 0):.2%}")
    print(f"Mean Stress: {validation.get('mean_stress', 0):.4f}")
    print(f"Continuity Preserved: {validation.get('continuity_preserved', False)}")

    # Save results
    results_output = {
        'experiment': '03-methodological-consistency',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'rho_default': RHO_DEFAULT,
            'k_max': K_MAX,
            'n_neighbors': N_NEIGHBORS,
            'training_n_max': TRAINING_N_MAX,
            'test_matrix': TEST_MATRIX
        },
        'scaling_data': {
            'N_values': scaling_stats['N_values'],
            'delta_means': scaling_stats['delta_means'],
            'delta_stds': scaling_stats['delta_stds'],
            'delta_ses': scaling_stats['delta_ses'],
            'delta_by_n': {str(k): v for k, v in scaling_stats['delta_by_n'].items()}
        },
        'validation': validation,
        'models': {
            'log_model': models['log_model'],
            'sqrt_model': models['sqrt_model'],
            'comparison': models['comparison']
        },
        'prediction_test': models['prediction_test'],
        'predictions': predictions,
        'summary': summary
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results_output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'metrics.json'}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(scaling_stats, models, predictions, output_dir)

    print("\nExperiment complete.")

    return results_output


if __name__ == '__main__':
    main()
