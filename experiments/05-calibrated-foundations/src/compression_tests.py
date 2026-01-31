"""
Phase 2: Compression tests.

Tests whether correlation modifications create measurable compression.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from lattice_generation import generate_2d_lattice, distances_to_correlation
from distance_transforms import apply_transform
from dimension_extraction import full_dimension_analysis


@dataclass
class CompressionResult:
    """Result from a single compression test."""
    condition: str
    d_topo: float
    d_corr: float
    delta: float
    valid: bool


def modify_correlation_global_boost(C: np.ndarray, epsilon: float = 0.3) -> np.ndarray:
    """
    Add uniform correlation boost to all pairs.

    C_new = (1 - epsilon) * C + epsilon

    This increases all correlations uniformly.
    """
    C_new = (1 - epsilon) * C + epsilon
    np.fill_diagonal(C_new, 1.0)
    return C_new


def modify_correlation_noise(C: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Add random noise to correlation matrix.

    Should NOT create systematic compression.
    """
    n = C.shape[0]
    noise = np.random.randn(n, n) * sigma
    noise = (noise + noise.T) / 2  # Symmetric
    np.fill_diagonal(noise, 0)

    C_new = C + noise
    # Clip to valid range
    C_new = np.clip(C_new, -1, 1)
    np.fill_diagonal(C_new, 1.0)
    return C_new


def modify_correlation_distance_decay(C: np.ndarray, D: np.ndarray,
                                       alpha: float = 0.1) -> np.ndarray:
    """
    Replace correlation with distance-decay function.

    C_new = exp(-alpha * D)

    Creates long-range correlations that decay with distance.
    """
    C_new = np.exp(-alpha * D)
    np.fill_diagonal(C_new, 1.0)
    return C_new


def run_compression_test(condition: str, C: np.ndarray, D_topo: np.ndarray,
                          transform: str, agreement_threshold: float = 0.5) -> CompressionResult:
    """
    Run a single compression test.

    Args:
        condition: Name of condition
        C: Correlation matrix
        D_topo: Topology distance matrix
        transform: Distance transform to use
        agreement_threshold: Method agreement threshold

    Returns:
        CompressionResult
    """
    D_corr = apply_transform(C, transform)
    result = full_dimension_analysis(D_topo, D_corr, agreement_threshold)

    return CompressionResult(
        condition=condition,
        d_topo=result['d_topo'] if result['valid'] else None,
        d_corr=result['d_corr'] if result['valid'] else None,
        delta=result['delta'] if result['valid'] else None,
        valid=result['valid']
    )


def run_compression_batch(condition: str, transform: str,
                          n_replications: int = 50,
                          side: int = 10,
                          modification_fn=None,
                          **mod_kwargs) -> Dict:
    """
    Run multiple compression test replications.

    Args:
        condition: Name of condition
        transform: Distance transform
        n_replications: Number of replications
        side: Lattice side length
        modification_fn: Function to modify correlation (or None for baseline)
        **mod_kwargs: Arguments for modification function

    Returns:
        Aggregated results
    """
    results = []
    valid_deltas = []

    for i in range(n_replications):
        # Generate base 2D lattice
        positions, D_topo, metadata = generate_2d_lattice(side)

        # Create base correlation
        C_base = distances_to_correlation(D_topo)

        # Apply modification if provided
        if modification_fn is not None:
            if 'D' in modification_fn.__code__.co_varnames:
                C = modification_fn(C_base, D_topo, **mod_kwargs)
            else:
                C = modification_fn(C_base, **mod_kwargs)
        else:
            C = C_base

        # Run test
        result = run_compression_test(condition, C, D_topo, transform)
        results.append(result)

        if result.valid and result.delta is not None:
            valid_deltas.append(result.delta)

    n_valid = len(valid_deltas)

    if n_valid > 0:
        delta_mean = np.mean(valid_deltas)
        delta_std = np.std(valid_deltas)
        delta_ci_low = np.percentile(valid_deltas, 2.5)
        delta_ci_high = np.percentile(valid_deltas, 97.5)
    else:
        delta_mean = delta_std = delta_ci_low = delta_ci_high = None

    return {
        'condition': condition,
        'n_replications': n_replications,
        'n_valid': n_valid,
        'delta_mean': delta_mean,
        'delta_std': delta_std,
        'delta_ci': (delta_ci_low, delta_ci_high),
        'all_deltas': valid_deltas,
        'results': results
    }


def evaluate_compression_predictions(results: Dict) -> Dict:
    """
    Evaluate Phase 2 predictions.

    Args:
        results: Dictionary with results for each condition

    Returns:
        Prediction evaluation results
    """
    predictions = {}

    # P2.1: Global boost creates compression (delta > 0.1)
    boost = results.get('boost', {})
    p21 = (boost.get('delta_mean') is not None and
           boost.get('delta_mean', 0) > 0.1)
    predictions['P2.1'] = {
        'description': 'Global boost creates compression',
        'threshold': 'delta > 0.1',
        'measured': boost.get('delta_mean'),
        'passed': p21
    }

    # P2.2: Noise control shows no compression (|delta| < 0.05)
    noise = results.get('noise', {})
    p22 = (noise.get('delta_mean') is not None and
           abs(noise.get('delta_mean', 1)) < 0.05)
    predictions['P2.2'] = {
        'description': 'Noise control shows no compression',
        'threshold': '|delta| < 0.05',
        'measured': noise.get('delta_mean'),
        'passed': p22
    }

    # P2.3: Distance-decay creates compression (delta > 0.05)
    decay = results.get('decay', {})
    p23 = (decay.get('delta_mean') is not None and
           decay.get('delta_mean', 0) > 0.05)
    predictions['P2.3'] = {
        'description': 'Distance-decay creates compression',
        'threshold': 'delta > 0.05',
        'measured': decay.get('delta_mean'),
        'passed': p23
    }

    # P2.4: Ordering: boost > decay > noise
    if all(r.get('delta_mean') is not None for r in [boost, decay, noise]):
        ordering = (boost['delta_mean'] > decay['delta_mean'] >
                    noise['delta_mean'])
    else:
        ordering = False

    predictions['P2.4'] = {
        'description': 'Effect ordering: boost > decay > noise',
        'threshold': 'boost > decay > noise',
        'measured': {
            'boost': boost.get('delta_mean'),
            'decay': decay.get('delta_mean'),
            'noise': noise.get('delta_mean')
        },
        'passed': ordering
    }

    return predictions


def run_full_compression_tests(transform: str,
                               n_replications: int = 50,
                               side: int = 10) -> Dict:
    """
    Run complete Phase 2 compression tests.

    Args:
        transform: Distance transform (from Phase 1)
        n_replications: Replications per condition
        side: Lattice side length

    Returns:
        Complete compression test results
    """
    print("\nPhase 2: Compression Tests")
    print("=" * 50)

    results = {}

    # Baseline (no modification)
    print("  Testing baseline...", end=" ")
    results['baseline'] = run_compression_batch(
        'baseline', transform, n_replications, side,
        modification_fn=None
    )
    print(f"delta = {results['baseline']['delta_mean']:.4f}")

    # Global boost
    print("  Testing global boost...", end=" ")
    results['boost'] = run_compression_batch(
        'boost', transform, n_replications, side,
        modification_fn=modify_correlation_global_boost,
        epsilon=0.3
    )
    print(f"delta = {results['boost']['delta_mean']:.4f}")

    # Random noise
    print("  Testing noise control...", end=" ")
    results['noise'] = run_compression_batch(
        'noise', transform, n_replications, side,
        modification_fn=modify_correlation_noise,
        sigma=0.1
    )
    print(f"delta = {results['noise']['delta_mean']:.4f}")

    # Distance decay
    print("  Testing distance decay...", end=" ")
    results['decay'] = run_compression_batch(
        'decay', transform, n_replications, side,
        modification_fn=modify_correlation_distance_decay,
        alpha=0.1
    )
    print(f"delta = {results['decay']['delta_mean']:.4f}")

    # Evaluate predictions
    print("\nEvaluating predictions...")
    predictions = evaluate_compression_predictions(results)

    for pred_id, pred in predictions.items():
        status = 'PASS' if pred['passed'] else 'FAIL'
        print(f"  {pred_id}: {pred['description']} - {status}")

    passed_count = sum(1 for p in predictions.values() if p['passed'])
    print(f"\n  Predictions passed: {passed_count}/4")

    return {
        'results': results,
        'predictions': predictions,
        'passed_count': passed_count
    }
