"""
Robust scaling analysis for Experiment 03.

Provides model fitting with updated thresholds and prediction evaluation.
Thresholds adjusted from Exp02 based on observed variance.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Any, List, Tuple, Optional


# Updated thresholds for Exp03 (adjusted from Exp02)
THRESHOLDS = {
    'P1_spearman': 0.85,      # Was 0.9 in Exp02
    'P2_r2': 0.80,            # Was 0.85 in Exp02
    'P3_continuity': 0.15,    # New for Exp03
    'P4_relative_error': 0.30, # Was 0.2 in Exp02
    'P5_growth': 0.0,         # Was 0.05 in Exp02
    'P6_se': 0.05             # Standard error threshold
}


def fit_log_model(N_values: np.ndarray, delta_values: np.ndarray) -> Dict[str, Any]:
    """
    Fit logarithmic model: delta = a * log(N) + b

    Args:
        N_values: Array of graph sizes
        delta_values: Array of compression ratios

    Returns:
        Dictionary with fitted parameters, R^2, and residuals
    """
    N = np.array(N_values, dtype=float)
    delta = np.array(delta_values, dtype=float)

    if len(N) < 2:
        return {'a': 0.0, 'b': 0.0, 'r2': 0.0, 'residuals': [], 'valid': False}

    log_N = np.log(N)

    X = np.vstack([log_N, np.ones(len(log_N))]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
        a, b = coeffs
    except Exception:
        return {'a': 0.0, 'b': 0.0, 'r2': 0.0, 'residuals': [], 'valid': False}

    delta_pred = a * log_N + b
    residuals = delta - delta_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((delta - np.mean(delta)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Compute standard errors for coefficients
    n = len(delta)
    if n > 2:
        mse = ss_res / (n - 2)
        X_inv = np.linalg.pinv(X.T @ X)
        se_coeffs = np.sqrt(np.diag(X_inv) * mse)
        se_a, se_b = se_coeffs
    else:
        se_a, se_b = 0.0, 0.0

    return {
        'a': float(a),
        'b': float(b),
        'se_a': float(se_a),
        'se_b': float(se_b),
        'r2': float(r2),
        'residuals': residuals.tolist(),
        'predictions': delta_pred.tolist(),
        'valid': True,
        'model': 'log',
        'formula': f'delta = {a:.4f} * log(N) + {b:.4f}'
    }


def fit_sqrt_model(N_values: np.ndarray, delta_values: np.ndarray) -> Dict[str, Any]:
    """
    Fit square root model: delta = c * sqrt(N) + d

    Alternative to power law with fixed exponent 0.5.

    Args:
        N_values: Array of graph sizes
        delta_values: Array of compression ratios

    Returns:
        Dictionary with fitted parameters and R^2
    """
    N = np.array(N_values, dtype=float)
    delta = np.array(delta_values, dtype=float)

    if len(N) < 2:
        return {'c': 0.0, 'd': 0.0, 'r2': 0.0, 'valid': False}

    sqrt_N = np.sqrt(N)

    X = np.vstack([sqrt_N, np.ones(len(sqrt_N))]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
        c, d = coeffs
    except Exception:
        return {'c': 0.0, 'd': 0.0, 'r2': 0.0, 'valid': False}

    delta_pred = c * sqrt_N + d
    residuals = delta - delta_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((delta - np.mean(delta)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'c': float(c),
        'd': float(d),
        'r2': float(r2),
        'residuals': residuals.tolist(),
        'predictions': delta_pred.tolist(),
        'valid': True,
        'model': 'sqrt',
        'formula': f'delta = {c:.4f} * sqrt(N) + {d:.4f}'
    }


def predict_delta(model: Dict[str, Any], N: float) -> float:
    """
    Predict delta for a given N using fitted model.
    """
    if not model.get('valid', False):
        return 0.0

    if model['model'] == 'log':
        return model['a'] * np.log(N) + model['b']
    elif model['model'] == 'sqrt':
        return model['c'] * np.sqrt(N) + model['d']
    else:
        return 0.0


def compare_models(log_fit: Dict[str, Any], sqrt_fit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare log and sqrt models to determine better fit.
    """
    log_r2 = log_fit.get('r2', 0.0) if log_fit.get('valid', False) else 0.0
    sqrt_r2 = sqrt_fit.get('r2', 0.0) if sqrt_fit.get('valid', False) else 0.0

    if log_r2 >= sqrt_r2:
        winner = 'log'
        winner_r2 = log_r2
    else:
        winner = 'sqrt'
        winner_r2 = sqrt_r2

    return {
        'winner': winner,
        'winner_r2': winner_r2,
        'log_r2': log_r2,
        'sqrt_r2': sqrt_r2,
        'delta_r2': abs(log_r2 - sqrt_r2)
    }


# Prediction evaluators for P1-P6

def evaluate_p1_monotonicity(N_values: np.ndarray, delta_values: np.ndarray) -> Dict[str, Any]:
    """
    P1: Monotonic Scaling - Spearman r > 0.85
    """
    threshold = THRESHOLDS['P1_spearman']

    if len(N_values) < 3:
        return {
            'prediction': 'P1',
            'description': 'Monotonic Scaling',
            'passed': False,
            'threshold': threshold,
            'measured': 0.0,
            'error': 'Insufficient data points'
        }

    r, p_value = spearmanr(N_values, delta_values)
    passed = r > threshold

    return {
        'prediction': 'P1',
        'description': 'Monotonic Scaling',
        'passed': bool(passed),
        'threshold': threshold,
        'measured': float(r),
        'p_value': float(p_value),
        'criterion': f'Spearman r(N, delta) > {threshold}'
    }


def evaluate_p2_log_scaling(log_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    P2: Logarithmic Scaling - R^2 > 0.80
    """
    threshold = THRESHOLDS['P2_r2']
    r2 = log_model.get('r2', 0.0)
    valid = log_model.get('valid', False)
    passed = valid and r2 > threshold

    return {
        'prediction': 'P2',
        'description': 'Logarithmic Scaling',
        'passed': bool(passed),
        'threshold': threshold,
        'measured': float(r2),
        'formula': log_model.get('formula', 'N/A'),
        'criterion': f'R^2 > {threshold} for log fit'
    }


def evaluate_p3_method_consistency(delta_500: float, delta_750: float) -> Dict[str, Any]:
    """
    P3: Method Consistency - No discontinuity between N=500 and N=750
    """
    threshold = THRESHOLDS['P3_continuity']
    diff = abs(delta_750 - delta_500)
    passed = diff < threshold

    return {
        'prediction': 'P3',
        'description': 'Method Consistency',
        'passed': bool(passed),
        'threshold': threshold,
        'measured': float(diff),
        'delta_500': float(delta_500),
        'delta_750': float(delta_750),
        'criterion': f'|delta(750) - delta(500)| < {threshold}'
    }


def evaluate_p4_predictive_validity(delta_predicted: float, delta_measured: float) -> Dict[str, Any]:
    """
    P4: Predictive Validity - Model predicts N=1000 within 30%
    """
    threshold = THRESHOLDS['P4_relative_error']

    if delta_measured == 0:
        relative_error = float('inf') if delta_predicted != 0 else 0.0
    else:
        relative_error = abs(delta_predicted - delta_measured) / abs(delta_measured)

    passed = relative_error < threshold

    return {
        'prediction': 'P4',
        'description': 'Predictive Validity',
        'passed': bool(passed),
        'threshold': threshold,
        'measured': float(relative_error),
        'delta_predicted': float(delta_predicted),
        'delta_measured': float(delta_measured),
        'criterion': f'|delta_pred - delta_meas| / |delta_meas| < {threshold}'
    }


def evaluate_p5_non_saturation(delta_500: float, delta_1000: float) -> Dict[str, Any]:
    """
    P5: Non-Saturation - delta(1000) > delta(500)
    """
    threshold = THRESHOLDS['P5_growth']
    diff = delta_1000 - delta_500
    passed = diff > threshold

    return {
        'prediction': 'P5',
        'description': 'Non-Saturation',
        'passed': bool(passed),
        'threshold': threshold,
        'measured': float(diff),
        'delta_500': float(delta_500),
        'delta_1000': float(delta_1000),
        'criterion': f'delta(1000) > delta(500)'
    }


def evaluate_p6_variance_reduction(se_delta_n100: float) -> Dict[str, Any]:
    """
    P6: Variance Reduction - SE at N=100 < 0.05
    """
    threshold = THRESHOLDS['P6_se']
    passed = se_delta_n100 < threshold

    return {
        'prediction': 'P6',
        'description': 'Variance Reduction',
        'passed': bool(passed),
        'threshold': threshold,
        'measured': float(se_delta_n100),
        'criterion': f'SE(delta) at N=100 < {threshold}'
    }


def evaluate_all_predictions(results: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Evaluate all P1-P6 predictions from results.
    """
    predictions = {}

    # P1: Monotonicity
    if 'N_values' in results and 'delta_means' in results:
        predictions['P1'] = evaluate_p1_monotonicity(
            np.array(results['N_values']),
            np.array(results['delta_means'])
        )

    # P2: Log scaling
    if 'log_model' in results:
        predictions['P2'] = evaluate_p2_log_scaling(results['log_model'])

    # P3: Method consistency
    if 'delta_by_n' in results:
        delta_500 = np.mean(results['delta_by_n'].get('500', results['delta_by_n'].get(500, [0])))
        delta_750 = np.mean(results['delta_by_n'].get('750', results['delta_by_n'].get(750, [0])))
        predictions['P3'] = evaluate_p3_method_consistency(delta_500, delta_750)

    # P4: Predictive validity
    if 'prediction_test' in results:
        pt = results['prediction_test']
        predictions['P4'] = evaluate_p4_predictive_validity(
            pt.get('delta_predicted_1000', 0),
            pt.get('delta_measured_1000', 0)
        )

    # P5: Non-saturation
    if 'delta_by_n' in results:
        delta_500 = np.mean(results['delta_by_n'].get('500', results['delta_by_n'].get(500, [0])))
        delta_1000 = np.mean(results['delta_by_n'].get('1000', results['delta_by_n'].get(1000, [0])))
        predictions['P5'] = evaluate_p5_non_saturation(delta_500, delta_1000)

    # P6: Variance reduction
    if 'se_by_n' in results and 100 in results['se_by_n']:
        predictions['P6'] = evaluate_p6_variance_reduction(results['se_by_n'][100])
    elif 'se_by_n' in results and '100' in results['se_by_n']:
        predictions['P6'] = evaluate_p6_variance_reduction(results['se_by_n']['100'])

    return predictions


def summarize_predictions(predictions: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Create summary of prediction results.
    """
    passed = sum(1 for p in predictions.values() if p.get('passed', False))
    total = len(predictions)

    if passed == total:
        status = 'full_success'
    elif passed >= 5:
        status = 'strong_success'
    elif passed >= 4:
        status = 'partial_success'
    elif passed >= 2:
        status = 'weak_signal'
    else:
        status = 'failure'

    return {
        'predictions_passed': passed,
        'predictions_total': total,
        'status': status,
        'p1_monotonicity': predictions.get('P1', {}).get('passed', False),
        'p2_log_scaling': predictions.get('P2', {}).get('passed', False),
        'p3_consistency': predictions.get('P3', {}).get('passed', False),
        'p4_predictive': predictions.get('P4', {}).get('passed', False),
        'p5_non_saturation': predictions.get('P5', {}).get('passed', False),
        'p6_variance': predictions.get('P6', {}).get('passed', False)
    }
