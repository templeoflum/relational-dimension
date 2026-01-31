"""
Scaling analysis module for Experiment 02.

Provides functions to fit log/power models to compression scaling data
and evaluate predictive validity.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from typing import Dict, Any, Tuple, List, Optional


def fit_log_model(N_values: np.ndarray, delta_values: np.ndarray) -> Dict[str, Any]:
    """
    Fit logarithmic model: δ = a * log(N) + b

    Args:
        N_values: Array of graph sizes
        delta_values: Array of compression ratios

    Returns:
        Dictionary with fitted parameters, R², and residuals
    """
    N = np.array(N_values, dtype=float)
    delta = np.array(delta_values, dtype=float)

    if len(N) < 2:
        return {'a': 0.0, 'b': 0.0, 'r2': 0.0, 'residuals': [], 'valid': False}

    # Log transform
    log_N = np.log(N)

    # Linear regression on log(N)
    # δ = a * log(N) + b
    X = np.vstack([log_N, np.ones(len(log_N))]).T
    try:
        coeffs, residuals_sum, rank, s = np.linalg.lstsq(X, delta, rcond=None)
        a, b = coeffs
    except Exception:
        return {'a': 0.0, 'b': 0.0, 'r2': 0.0, 'residuals': [], 'valid': False}

    # Predictions and R²
    delta_pred = a * log_N + b
    residuals = delta - delta_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((delta - np.mean(delta)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'a': float(a),
        'b': float(b),
        'r2': float(r2),
        'residuals': residuals.tolist(),
        'predictions': delta_pred.tolist(),
        'valid': True,
        'model': 'log',
        'formula': f'd = {a:.4f} * log(N) + {b:.4f}'
    }


def fit_power_model(N_values: np.ndarray, delta_values: np.ndarray) -> Dict[str, Any]:
    """
    Fit power law model: δ = c * N^α

    Uses log-log linear regression: log(δ) = α * log(N) + log(c)

    Args:
        N_values: Array of graph sizes
        delta_values: Array of compression ratios

    Returns:
        Dictionary with fitted parameters, R², and residuals
    """
    N = np.array(N_values, dtype=float)
    delta = np.array(delta_values, dtype=float)

    if len(N) < 2:
        return {'c': 0.0, 'alpha': 0.0, 'r2': 0.0, 'residuals': [], 'valid': False}

    # Filter out non-positive delta values for log transform
    mask = delta > 0
    if np.sum(mask) < 2:
        return {'c': 0.0, 'alpha': 0.0, 'r2': 0.0, 'residuals': [], 'valid': False}

    N_pos = N[mask]
    delta_pos = delta[mask]

    # Log-log transform
    log_N = np.log(N_pos)
    log_delta = np.log(delta_pos)

    # Linear regression on log-log
    # log(δ) = α * log(N) + log(c)
    X = np.vstack([log_N, np.ones(len(log_N))]).T
    try:
        coeffs, residuals_sum, rank, s = np.linalg.lstsq(X, log_delta, rcond=None)
        alpha, log_c = coeffs
        c = np.exp(log_c)
    except Exception:
        return {'c': 0.0, 'alpha': 0.0, 'r2': 0.0, 'residuals': [], 'valid': False}

    # Predictions in original space
    delta_pred_all = c * (N ** alpha)
    residuals = delta - delta_pred_all

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((delta - np.mean(delta)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'c': float(c),
        'alpha': float(alpha),
        'r2': float(r2),
        'residuals': residuals.tolist(),
        'predictions': delta_pred_all.tolist(),
        'valid': True,
        'model': 'power',
        'formula': f'd = {c:.4f} * N^{alpha:.4f}'
    }


def predict_delta(model: Dict[str, Any], N: float) -> float:
    """
    Predict delta for a given N using fitted model.

    Args:
        model: Fitted model dictionary (from fit_log_model or fit_power_model)
        N: Graph size to predict for

    Returns:
        Predicted delta value
    """
    if not model.get('valid', False):
        return 0.0

    if model['model'] == 'log':
        return model['a'] * np.log(N) + model['b']
    elif model['model'] == 'power':
        return model['c'] * (N ** model['alpha'])
    else:
        return 0.0


def compare_models(log_fit: Dict[str, Any], power_fit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare log and power models to determine better fit.

    Args:
        log_fit: Fitted log model
        power_fit: Fitted power model

    Returns:
        Dictionary with comparison results
    """
    log_r2 = log_fit.get('r2', 0.0) if log_fit.get('valid', False) else 0.0
    power_r2 = power_fit.get('r2', 0.0) if power_fit.get('valid', False) else 0.0

    # Determine winner
    if log_r2 > power_r2:
        winner = 'log'
        winner_r2 = log_r2
        delta_r2 = log_r2 - power_r2
    elif power_r2 > log_r2:
        winner = 'power'
        winner_r2 = power_r2
        delta_r2 = power_r2 - log_r2
    else:
        winner = 'tie'
        winner_r2 = log_r2
        delta_r2 = 0.0

    return {
        'winner': winner,
        'winner_r2': winner_r2,
        'log_r2': log_r2,
        'power_r2': power_r2,
        'delta_r2': delta_r2,
        'log_valid': log_fit.get('valid', False),
        'power_valid': power_fit.get('valid', False)
    }


def evaluate_monotonicity(N_values: np.ndarray, delta_values: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate P1: Monotonic scaling.

    Args:
        N_values: Array of graph sizes
        delta_values: Array of compression ratios (mean per N)

    Returns:
        Prediction evaluation result
    """
    if len(N_values) < 3:
        return {
            'prediction': 'P1',
            'passed': False,
            'spearman_r': 0.0,
            'p_value': 1.0,
            'error': 'Insufficient data points'
        }

    r, p_value = spearmanr(N_values, delta_values)

    passed = r > 0.9

    return {
        'prediction': 'P1',
        'description': 'Monotonic Scaling',
        'passed': bool(passed),
        'threshold': 0.9,
        'measured': float(r),
        'p_value': float(p_value),
        'criterion': 'Spearman r(N, δ) > 0.9'
    }


def evaluate_log_fit(log_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate P2: Logarithmic functional form.

    Args:
        log_model: Fitted log model

    Returns:
        Prediction evaluation result
    """
    r2 = log_model.get('r2', 0.0)
    valid = log_model.get('valid', False)

    passed = valid and r2 > 0.85

    return {
        'prediction': 'P2',
        'description': 'Logarithmic Functional Form',
        'passed': bool(passed),
        'threshold': 0.85,
        'measured': float(r2),
        'formula': log_model.get('formula', 'N/A'),
        'criterion': 'R² > 0.85 for log fit'
    }


def evaluate_power_fit(power_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate P3: Power law functional form.

    Args:
        power_model: Fitted power model

    Returns:
        Prediction evaluation result
    """
    r2 = power_model.get('r2', 0.0)
    alpha = power_model.get('alpha', 0.0)
    valid = power_model.get('valid', False)

    passed = valid and r2 > 0.85 and alpha > 0

    return {
        'prediction': 'P3',
        'description': 'Power Law Functional Form',
        'passed': bool(passed),
        'threshold': 0.85,
        'measured': float(r2),
        'alpha': float(alpha),
        'formula': power_model.get('formula', 'N/A'),
        'criterion': 'R² > 0.85 for power fit, α > 0'
    }


def evaluate_saturation(delta_500: float, delta_1000: float) -> Dict[str, Any]:
    """
    Evaluate P4: Non-saturation.

    Args:
        delta_500: Mean delta at N=500
        delta_1000: Mean delta at N=1000

    Returns:
        Prediction evaluation result
    """
    difference = delta_1000 - delta_500
    passed = difference > 0.05

    return {
        'prediction': 'P4',
        'description': 'Non-Saturation',
        'passed': bool(passed),
        'threshold': 0.05,
        'measured': float(difference),
        'delta_500': float(delta_500),
        'delta_1000': float(delta_1000),
        'criterion': 'δ(1000) > δ(500) + 0.05'
    }


def evaluate_predictive_validity(delta_predicted: float, delta_measured: float) -> Dict[str, Any]:
    """
    Evaluate P5: Predictive validity.

    Args:
        delta_predicted: Predicted delta at N=1000 (from model fitted on N<=500)
        delta_measured: Measured delta at N=1000

    Returns:
        Prediction evaluation result
    """
    if delta_measured == 0:
        relative_error = float('inf') if delta_predicted != 0 else 0.0
    else:
        relative_error = abs(delta_predicted - delta_measured) / abs(delta_measured)

    passed = relative_error < 0.2

    return {
        'prediction': 'P5',
        'description': 'Predictive Validity',
        'passed': bool(passed),
        'threshold': 0.2,
        'measured': float(relative_error),
        'delta_predicted': float(delta_predicted),
        'delta_measured': float(delta_measured),
        'criterion': '|δ_predicted - δ_measured| / δ_measured < 0.2'
    }


def evaluate_variance_reduction(std_delta_n100: float) -> Dict[str, Any]:
    """
    Evaluate P6: Variance reduction with improved method.

    Args:
        std_delta_n100: Standard deviation of delta at N=100

    Returns:
        Prediction evaluation result
    """
    passed = std_delta_n100 < 0.15

    return {
        'prediction': 'P6',
        'description': 'Improved Baseline Variance',
        'passed': bool(passed),
        'threshold': 0.15,
        'measured': float(std_delta_n100),
        'criterion': 'Std(δ) at N=100 < 0.15'
    }


def evaluate_all_predictions(scaling_results: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Evaluate all P1-P6 predictions from scaling results.

    Args:
        scaling_results: Full results dictionary from scaling experiment

    Returns:
        Dictionary of prediction evaluations
    """
    predictions = {}

    # P1: Monotonicity
    if 'N_values' in scaling_results and 'delta_means' in scaling_results:
        predictions['P1'] = evaluate_monotonicity(
            np.array(scaling_results['N_values']),
            np.array(scaling_results['delta_means'])
        )

    # P2: Log fit
    if 'log_model' in scaling_results:
        predictions['P2'] = evaluate_log_fit(scaling_results['log_model'])

    # P3: Power fit
    if 'power_model' in scaling_results:
        predictions['P3'] = evaluate_power_fit(scaling_results['power_model'])

    # P4: Saturation
    if 'delta_by_n' in scaling_results:
        delta_500 = np.mean(scaling_results['delta_by_n'].get(500, [0]))
        delta_1000 = np.mean(scaling_results['delta_by_n'].get(1000, [0]))
        predictions['P4'] = evaluate_saturation(delta_500, delta_1000)

    # P5: Predictive validity
    if 'delta_predicted_1000' in scaling_results and 'delta_measured_1000' in scaling_results:
        predictions['P5'] = evaluate_predictive_validity(
            scaling_results['delta_predicted_1000'],
            scaling_results['delta_measured_1000']
        )

    # P6: Variance reduction
    if 'std_by_n' in scaling_results and 100 in scaling_results['std_by_n']:
        predictions['P6'] = evaluate_variance_reduction(
            scaling_results['std_by_n'][100]
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

    # Determine which functional form is validated
    p2_passed = predictions.get('P2', {}).get('passed', False)
    p3_passed = predictions.get('P3', {}).get('passed', False)

    if p2_passed and p3_passed:
        functional_form = 'both'
    elif p2_passed:
        functional_form = 'logarithmic'
    elif p3_passed:
        functional_form = 'power_law'
    else:
        functional_form = 'neither'

    # Check P1 (core hypothesis for this experiment)
    p1_passed = predictions.get('P1', {}).get('passed', False)

    if passed == total:
        status = 'success'
    elif passed >= total / 2 and p1_passed:
        status = 'partial'
    elif not p1_passed:
        status = 'falsified'
    else:
        status = 'partial'

    return {
        'predictions_passed': passed,
        'predictions_total': total,
        'status': status,
        'monotonic_scaling_passed': p1_passed,
        'functional_form': functional_form,
        'predictive_validity': predictions.get('P5', {}).get('passed', False)
    }
