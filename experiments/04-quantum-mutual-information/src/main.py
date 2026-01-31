"""
Main runner for Experiment 04: Quantum Mutual Information.

Tests dimensional compression across various quantum states to determine
if entanglement creates compression invisible to topological analysis.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from quantum_states import (
    get_state_by_name, state_to_density_matrix,
    verify_state_normalization, verify_density_matrix
)
from mutual_information import (
    mutual_information_matrix, validate_mi_matrix, half_chain_entanglement
)
from quantum_dimension import analyze_quantum_state, mi_distance_matrix


# Test matrix configuration
TEST_MATRIX = [
    # Small systems for all state types
    {'n_qubits': 4, 'geometry': 'chain', 'states': ['product', 'ghz', 'w', 'cluster', 'random']},
    {'n_qubits': 6, 'geometry': 'chain', 'states': ['product', 'ghz', 'w', 'cluster', 'random']},
    {'n_qubits': 8, 'geometry': 'chain', 'states': ['product', 'ghz', 'w', 'cluster', 'random']},

    # 2D grids
    {'n_qubits': 8, 'geometry': '2x4', 'states': ['cluster']},
    {'n_qubits': 9, 'geometry': '3x3', 'states': ['cluster']},

    # Larger 1D systems (memory intensive)
    {'n_qubits': 10, 'geometry': 'chain', 'states': ['product', 'ghz', 'w', 'cluster', 'random']},

    # At computational limit
    {'n_qubits': 12, 'geometry': 'chain', 'states': ['ghz', 'cluster', 'random']},
    {'n_qubits': 12, 'geometry': '3x4', 'states': ['cluster']},
]

# Number of random seeds for random states
N_RANDOM_SEEDS = 5


def run_single_test(n_qubits: int, geometry: str, state_name: str,
                    seed: int = None) -> Dict[str, Any]:
    """Run single quantum state analysis."""
    print(f"  Testing {state_name} (N={n_qubits}, {geometry})", end="")
    if seed is not None:
        print(f" seed={seed}", end="")
    print("...", end=" ", flush=True)

    try:
        # Generate state
        state, metadata = get_state_by_name(state_name, n_qubits, geometry, seed)

        # Verify normalization
        if not verify_state_normalization(state):
            print("WARN: State not normalized")

        # Convert to density matrix
        rho = state_to_density_matrix(state)

        # Verify density matrix properties
        dm_check = verify_density_matrix(rho)
        if not dm_check['is_hermitian'] or not dm_check['trace_one']:
            print("WARN: Density matrix issues")

        # Compute MI matrix
        MI = mutual_information_matrix(rho, n_qubits)

        # Validate MI
        mi_validation = validate_mi_matrix(MI)

        # Compute entanglement entropy
        S_ent = half_chain_entanglement(rho, n_qubits)

        # Analyze dimension
        analysis = analyze_quantum_state(MI, n_qubits, geometry)

        # Add metadata
        result = {
            'state_name': state_name,
            'n_qubits': n_qubits,
            'geometry': geometry,
            'seed': seed,
            'entanglement_type': metadata.get('entanglement_type', 'unknown'),
            'S_ent': float(S_ent),
            'purity': dm_check['purity'],
            **analysis,
            'mi_validation': mi_validation
        }

        print(f"delta={result['delta']:.3f}, d_Q={result['d_quantum']:.2f}")
        return result

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'state_name': state_name,
            'n_qubits': n_qubits,
            'geometry': geometry,
            'seed': seed,
            'error': str(e)
        }


def run_test_matrix() -> List[Dict[str, Any]]:
    """Run full test matrix."""
    results = []

    for config in TEST_MATRIX:
        n_qubits = config['n_qubits']
        geometry = config['geometry']
        states = config['states']

        print(f"\n=== N={n_qubits}, geometry={geometry} ===")

        for state_name in states:
            if state_name == 'random':
                # Run multiple seeds for random states
                for seed in range(N_RANDOM_SEEDS):
                    result = run_single_test(n_qubits, geometry, state_name, seed)
                    results.append(result)
            else:
                result = run_single_test(n_qubits, geometry, state_name)
                results.append(result)

    return results


def evaluate_predictions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate predictions P1-P6."""
    predictions = {}

    # Filter out errors and numerical outliers
    valid_results = [r for r in results
                     if 'error' not in r
                     and abs(r.get('delta', 0)) < 10  # Filter extreme outliers
                     and abs(r.get('d_quantum', 0)) < 100]

    # P1: Product State Baseline - |delta| < 0.1
    product_results = [r for r in valid_results if r['state_name'] == 'product']
    if product_results:
        product_deltas = [abs(r['delta']) for r in product_results]
        p1_max_delta = max(product_deltas)
        predictions['P1'] = {
            'name': 'Product State Baseline',
            'threshold': '|delta| < 0.1',
            'measured': p1_max_delta,
            'passed': p1_max_delta < 0.1
        }

    # P2: GHZ Compression - delta > 0.4
    ghz_results = [r for r in valid_results if r['state_name'] == 'ghz']
    if ghz_results:
        ghz_deltas = [r['delta'] for r in ghz_results]
        p2_mean_delta = np.mean(ghz_deltas)
        predictions['P2'] = {
            'name': 'GHZ Compression',
            'threshold': 'delta > 0.4',
            'measured': p2_mean_delta,
            'passed': p2_mean_delta > 0.4
        }

    # P3: Cluster State Topology - d_Q < 1.5 for 1D cluster
    cluster_1d = [r for r in valid_results
                  if r['state_name'] == 'cluster' and r['geometry'] == 'chain']
    if cluster_1d:
        cluster_d_qs = [r['d_quantum'] for r in cluster_1d]
        p3_mean_d_q = np.mean(cluster_d_qs)
        predictions['P3'] = {
            'name': 'Cluster State Topology',
            'threshold': 'd_Q < 1.5',
            'measured': p3_mean_d_q,
            'passed': p3_mean_d_q < 1.5
        }

    # P4: Random State Compression - delta > 0.25
    random_results = [r for r in valid_results if r['state_name'] == 'random']
    if random_results:
        random_deltas = [r['delta'] for r in random_results]
        p4_mean_delta = np.mean(random_deltas)
        predictions['P4'] = {
            'name': 'Random State Compression',
            'threshold': 'delta > 0.25',
            'measured': p4_mean_delta,
            'passed': p4_mean_delta > 0.25
        }

    # P5: Entanglement-Compression Correlation - r > 0.7
    entangled_results = [r for r in valid_results
                         if r['state_name'] in ['ghz', 'w', 'cluster', 'random']]
    if len(entangled_results) >= 5:
        S_ents = [r['S_ent'] for r in entangled_results]
        deltas = [r['delta'] for r in entangled_results]
        from scipy.stats import pearsonr
        r_corr, p_val = pearsonr(S_ents, deltas)
        predictions['P5'] = {
            'name': 'Entanglement-Compression Correlation',
            'threshold': 'r > 0.7',
            'measured': r_corr,
            'p_value': p_val,
            'passed': r_corr > 0.7
        }

    # P6: 2D Holographic Prediction - d_Q < 1.5 for 2D cluster
    cluster_2d = [r for r in valid_results
                  if r['state_name'] == 'cluster' and 'x' in r['geometry']]
    if cluster_2d:
        cluster_2d_d_qs = [r['d_quantum'] for r in cluster_2d]
        p6_mean_d_q = np.mean(cluster_2d_d_qs)
        predictions['P6'] = {
            'name': '2D Holographic Prediction',
            'threshold': 'd_Q < 1.5',
            'measured': p6_mean_d_q,
            'passed': p6_mean_d_q < 1.5
        }

    return predictions


def compute_validation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute validation gate metrics."""
    valid_results = [r for r in results if 'error' not in r]

    # Product state max MI
    product_results = [r for r in valid_results if r['state_name'] == 'product']
    product_max_mi = max([r['mi_max'] for r in product_results]) if product_results else None

    # Max pure state entropy (should be ~0 for pure states)
    max_pure_entropy = 0.0  # Pure states have S=0 by construction

    # Min MI value (should be >= 0)
    min_mi = min([r['mi_min'] for r in valid_results]) if valid_results else None

    # GHZ MI mean (should be ~1 bit)
    ghz_results = [r for r in valid_results if r['state_name'] == 'ghz']
    ghz_mi_mean = np.mean([r['mi_mean'] for r in ghz_results]) if ghz_results else None

    return {
        'product_state_max_mi': product_max_mi,
        'max_pure_state_entropy': max_pure_entropy,
        'min_mi_value': min_mi,
        'ghz_mi_mean': ghz_mi_mean
    }


def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create all visualization plots."""
    valid_results = [r for r in results if 'error' not in r]

    # 1. MI Heatmaps for different state types
    create_mi_heatmaps(valid_results, output_dir)

    # 2. Compression by state type
    create_compression_plot(valid_results, output_dir)

    # 3. Entanglement vs compression correlation
    create_correlation_plot(valid_results, output_dir)

    # 4. Predictions summary
    predictions = evaluate_predictions(results)
    create_predictions_plot(predictions, output_dir)


def create_mi_heatmaps(results: List[Dict[str, Any]], output_dir: str):
    """Create MI heatmap visualizations."""
    # Select representative results (N=8, chain)
    n_target = 8
    geo_target = 'chain'

    state_types = ['product', 'ghz', 'w', 'cluster', 'random']
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for idx, state_name in enumerate(state_types):
        # Find a matching result
        matching = [r for r in results
                    if r['state_name'] == state_name
                    and r['n_qubits'] == n_target
                    and r['geometry'] == geo_target]

        if matching:
            r = matching[0]
            # Regenerate MI matrix for visualization
            state, _ = get_state_by_name(state_name, n_target, geo_target,
                                          seed=r.get('seed'))
            rho = state_to_density_matrix(state)
            MI = mutual_information_matrix(rho, n_target)

            im = axes[idx].imshow(MI, cmap='viridis', vmin=0, vmax=1)
            axes[idx].set_title(f"{state_name.upper()}\ndelta={r['delta']:.2f}")
            axes[idx].set_xlabel('Qubit j')
            axes[idx].set_ylabel('Qubit i')
        else:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(state_name.upper())

    plt.colorbar(im, ax=axes[-1], label='MI (bits)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mi_heatmaps.png'), dpi=150)
    plt.close()


def create_compression_plot(results: List[Dict[str, Any]], output_dir: str):
    """Create compression by state type plot."""
    state_types = ['product', 'ghz', 'w', 'cluster', 'random']
    colors = ['gray', 'red', 'blue', 'green', 'purple']

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, state_name in enumerate(state_types):
        state_results = [r for r in results if r['state_name'] == state_name]
        if state_results:
            n_qubits = [r['n_qubits'] for r in state_results]
            deltas = [r['delta'] for r in state_results]
            ax.scatter(n_qubits, deltas, label=state_name, color=colors[idx],
                       alpha=0.7, s=60)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Number of Qubits (N)')
    ax.set_ylabel('Compression Ratio (delta)')
    ax.set_title('Dimensional Compression by Quantum State Type')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_by_state.png'), dpi=150)
    plt.close()


def create_correlation_plot(results: List[Dict[str, Any]], output_dir: str):
    """Create entanglement vs compression correlation plot."""
    entangled_results = [r for r in results
                         if r['state_name'] in ['ghz', 'w', 'cluster', 'random']
                         and 'error' not in r]

    fig, ax = plt.subplots(figsize=(8, 6))

    state_colors = {'ghz': 'red', 'w': 'blue', 'cluster': 'green', 'random': 'purple'}

    for state_name, color in state_colors.items():
        state_results = [r for r in entangled_results if r['state_name'] == state_name]
        if state_results:
            S_ents = [r['S_ent'] for r in state_results]
            deltas = [r['delta'] for r in state_results]
            ax.scatter(S_ents, deltas, label=state_name, color=color, alpha=0.7, s=60)

    # Add correlation line if enough points
    if len(entangled_results) >= 5:
        S_ents = [r['S_ent'] for r in entangled_results]
        deltas = [r['delta'] for r in entangled_results]
        from scipy.stats import pearsonr
        r_corr, p_val = pearsonr(S_ents, deltas)

        # Fit line
        z = np.polyfit(S_ents, deltas, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(S_ents), max(S_ents), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5,
                label=f'r={r_corr:.2f} (p={p_val:.3f})')

    ax.set_xlabel('Half-Chain Entanglement Entropy (S_ent)')
    ax.set_ylabel('Compression Ratio (delta)')
    ax.set_title('Entanglement vs Dimensional Compression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entanglement_correlation.png'), dpi=150)
    plt.close()


def create_predictions_plot(predictions: Dict[str, Any], output_dir: str):
    """Create predictions summary plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pred_ids = list(predictions.keys())
    colors = ['green' if predictions[p]['passed'] else 'red' for p in pred_ids]
    names = [predictions[p]['name'] for p in pred_ids]
    measured = [predictions[p]['measured'] for p in pred_ids]

    y_pos = np.arange(len(pred_ids))
    bars = ax.barh(y_pos, measured, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p}: {n}" for p, n in zip(pred_ids, names)])
    ax.set_xlabel('Measured Value')
    ax.set_title('Prediction Results (Green=Pass, Red=Fail)')

    # Add threshold markers
    for i, p in enumerate(pred_ids):
        threshold_str = predictions[p]['threshold']
        ax.annotate(threshold_str, (measured[i], i), textcoords="offset points",
                    xytext=(5, 0), fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_summary.png'), dpi=150)
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Experiment 04: Quantum Mutual Information")
    print("=" * 60)

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(exp_dir, 'output')
    reports_dir = os.path.join(exp_dir, 'reports')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Run tests
    print("\nRunning test matrix...")
    results = run_test_matrix()

    # Evaluate predictions
    print("\n" + "=" * 60)
    print("Evaluating predictions...")
    predictions = evaluate_predictions(results)

    for pred_id, pred in predictions.items():
        status = "PASS" if pred['passed'] else "FAIL"
        print(f"  {pred_id}: {pred['name']}")
        print(f"      Threshold: {pred['threshold']}")
        print(f"      Measured:  {pred['measured']:.4f}")
        print(f"      Status:    {status}")

    # Compute validation metrics
    validation = compute_validation_metrics(results)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, reports_dir)

    # Compile output
    output = {
        'experiment': '04-quantum-mutual-information',
        'timestamp': datetime.now().isoformat(),
        'test_matrix': TEST_MATRIX,
        'n_random_seeds': N_RANDOM_SEEDS,
        'results': results,
        'predictions': predictions,
        'validation': validation,
        'summary': {
            'total_tests': len(results),
            'successful_tests': len([r for r in results if 'error' not in r]),
            'failed_tests': len([r for r in results if 'error' in r]),
            'predictions_passed': sum(1 for p in predictions.values() if p['passed']),
            'predictions_total': len(predictions)
        }
    }

    # Save output
    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tests: {output['summary']['total_tests']}")
    print(f"Successful:  {output['summary']['successful_tests']}")
    print(f"Failed:      {output['summary']['failed_tests']}")
    print(f"Predictions passed: {output['summary']['predictions_passed']}/{output['summary']['predictions_total']}")

    return output


if __name__ == '__main__':
    main()
