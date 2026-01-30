"""
Metrics module for Experiment 01.

Provides functions to compute compression ratios, aggregate results,
and evaluate predictions against thresholds.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import pearsonr


def compression_ratio(d_topo: float, d_corr: float) -> float:
    """
    Compute compression ratio delta.

    delta = (d_topo - d_corr) / d_topo

    Args:
        d_topo: Topological dimension
        d_corr: Correlation dimension

    Returns:
        Compression ratio (positive = compression, negative = expansion)
    """
    if d_topo == 0:
        return 0.0
    return (d_topo - d_corr) / d_topo


def aggregate_results(results_list: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate results from multiple replications.

    Args:
        results_list: List of result dictionaries from individual runs

    Returns:
        Aggregated statistics (mean, std, min, max for each metric)
    """
    if not results_list:
        return {}

    # Get all numeric keys from first result
    keys = [k for k in results_list[0].keys()
            if isinstance(results_list[0][k], (int, float, np.number))]

    aggregated = {}
    for key in keys:
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))

    aggregated['n_replications'] = len(results_list)

    return aggregated


def evaluate_p1(delta_values: List[float], threshold: float = 0.2) -> Dict[str, Any]:
    """
    Evaluate P1: Baseline Agreement.

    For NN correlations, |d_topo - d_corr| should be < threshold.
    Since delta = (d_topo - d_corr) / d_topo, we check |delta * d_topo| < threshold.
    But simpler: for baseline, delta itself should be near 0.

    Args:
        delta_values: Compression ratios from NN configurations
        threshold: Maximum allowed |delta| (as fraction, so threshold=0.2 means 20%)

    Returns:
        Prediction evaluation result
    """
    mean_abs_delta = float(np.mean(np.abs(delta_values)))
    passed = mean_abs_delta < threshold

    return {
        'prediction': 'P1',
        'description': 'Baseline Agreement',
        'passed': passed,
        'threshold': threshold,
        'measured': mean_abs_delta,
        'criterion': f'|delta| < {threshold}'
    }


def evaluate_p2(delta_values: List[float], threshold: float = 0.25) -> Dict[str, Any]:
    """
    Evaluate P2: Long-Range Compression.

    For LR correlations, delta should be > threshold.

    Args:
        delta_values: Compression ratios from LR configurations
        threshold: Minimum required delta

    Returns:
        Prediction evaluation result
    """
    mean_delta = float(np.mean(delta_values))
    passed = mean_delta > threshold

    return {
        'prediction': 'P2',
        'description': 'Long-Range Compression',
        'passed': passed,
        'threshold': threshold,
        'measured': mean_delta,
        'criterion': f'delta > {threshold}'
    }


def evaluate_p3(delta_small: List[float], delta_large: List[float],
                threshold: float = 1.3) -> Dict[str, Any]:
    """
    Evaluate P3: Scaling Behavior.

    Compression should increase with graph size: delta_200 / delta_50 > threshold.

    Args:
        delta_small: Compression ratios for small graphs (N=50)
        delta_large: Compression ratios for large graphs (N=200)
        threshold: Minimum ratio required

    Returns:
        Prediction evaluation result
    """
    mean_small = float(np.mean(delta_small))
    mean_large = float(np.mean(delta_large))

    if mean_small == 0:
        ratio = float('inf') if mean_large > 0 else 1.0
    else:
        ratio = mean_large / mean_small

    passed = ratio > threshold

    return {
        'prediction': 'P3',
        'description': 'Scaling Behavior',
        'passed': passed,
        'threshold': threshold,
        'measured': ratio,
        'delta_small_mean': mean_small,
        'delta_large_mean': mean_large,
        'criterion': f'delta_200 / delta_50 > {threshold}'
    }


def evaluate_p4(delta_random: List[float], delta_lr: List[float],
                threshold_random: float = 0.15,
                threshold_lr: float = 0.25) -> Dict[str, Any]:
    """
    Evaluate P4: Random Control.

    Random correlations should not produce compression: delta_random < threshold_random.
    AND LR should still show compression: delta_lr > threshold_lr.

    Args:
        delta_random: Compression ratios for random correlations
        delta_lr: Compression ratios for LR correlations
        threshold_random: Maximum delta for random (should be small)
        threshold_lr: Minimum delta for LR (should be significant)

    Returns:
        Prediction evaluation result
    """
    mean_random = float(np.mean(delta_random))
    mean_lr = float(np.mean(delta_lr))

    random_ok = mean_random < threshold_random
    lr_ok = mean_lr > threshold_lr
    passed = random_ok and lr_ok

    return {
        'prediction': 'P4',
        'description': 'Random Control',
        'passed': passed,
        'threshold_random': threshold_random,
        'threshold_lr': threshold_lr,
        'measured_random': mean_random,
        'measured_lr': mean_lr,
        'criterion': f'delta_random < {threshold_random} AND delta_LR > {threshold_lr}'
    }


def evaluate_p5(alpha_values: List[float], delta_values: List[float],
                threshold: float = 0.8) -> Dict[str, Any]:
    """
    Evaluate P5: Dose-Response.

    Compression should correlate with correlation strength: r(alpha, delta) > threshold.

    Args:
        alpha_values: Correlation strength values
        delta_values: Corresponding compression ratios
        threshold: Minimum Pearson correlation required

    Returns:
        Prediction evaluation result
    """
    if len(alpha_values) < 3:
        return {
            'prediction': 'P5',
            'description': 'Dose-Response',
            'passed': False,
            'threshold': threshold,
            'measured': 0.0,
            'error': 'Insufficient data points'
        }

    r, p_value = pearsonr(alpha_values, delta_values)
    passed = r > threshold

    return {
        'prediction': 'P5',
        'description': 'Dose-Response',
        'passed': passed,
        'threshold': threshold,
        'measured': float(r),
        'p_value': float(p_value),
        'criterion': f'r(alpha, delta) > {threshold}'
    }


def evaluate_all_predictions(results: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Evaluate all predictions from full results.

    Args:
        results: Full results dictionary from experiment

    Returns:
        Dictionary of prediction evaluations
    """
    predictions = {}

    # P1: Baseline - use NN results
    if 'rgg_nn' in results and 'delta_all' in results['rgg_nn']:
        predictions['P1'] = evaluate_p1(results['rgg_nn']['delta_all'])

    # P2: Core - use LR results
    if 'rgg_lr' in results and 'delta_all' in results['rgg_lr']:
        predictions['P2'] = evaluate_p2(results['rgg_lr']['delta_all'])

    # P3: Scaling - compare small vs large
    if 'rgg_lr' in results:
        lr_data = results['rgg_lr']
        if 'delta_by_n' in lr_data:
            small_n = min(lr_data['delta_by_n'].keys())
            large_n = max(lr_data['delta_by_n'].keys())
            predictions['P3'] = evaluate_p3(
                lr_data['delta_by_n'][small_n],
                lr_data['delta_by_n'][large_n]
            )

    # P4: Control - compare random vs LR
    if 'rgg_rand' in results and 'rgg_lr' in results:
        predictions['P4'] = evaluate_p4(
            results['rgg_rand'].get('delta_all', []),
            results['rgg_lr'].get('delta_all', [])
        )

    # P5: Dose-response
    if 'dose_response' in results:
        dr = results['dose_response']
        predictions['P5'] = evaluate_p5(
            dr.get('alpha_values', []),
            dr.get('delta_values', [])
        )

    return predictions


def summarize_predictions(predictions: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Create summary of prediction results.

    Args:
        predictions: Dictionary of prediction evaluations

    Returns:
        Summary with counts and overall status
    """
    passed = sum(1 for p in predictions.values() if p.get('passed', False))
    total = len(predictions)

    if passed == total:
        status = 'success'
    elif passed >= total / 2:
        status = 'partial'
    else:
        status = 'falsified'

    # Check specifically if P2 (core hypothesis) passed
    if 'P2' in predictions and not predictions['P2'].get('passed', False):
        status = 'falsified'

    return {
        'predictions_passed': passed,
        'predictions_total': total,
        'status': status,
        'core_hypothesis_passed': predictions.get('P2', {}).get('passed', False)
    }
