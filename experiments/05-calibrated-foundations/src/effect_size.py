"""
Phase 3: Effect size estimation.

Computes Cohen's d, confidence intervals, and required sample sizes.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.

    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return abs(d)


def cohens_d_one_sample(data: List[float], null_value: float = 0.0) -> float:
    """
    Compute one-sample Cohen's d.

    d = (mean - null_value) / std
    """
    if len(data) < 2:
        return 0.0

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std < 1e-10:
        return 0.0

    return abs(mean - null_value) / std


def bootstrap_ci(data: List[float], confidence: float = 0.95,
                 n_bootstrap: int = 10000) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Returns:
        (lower, upper) confidence interval
    """
    if len(data) < 2:
        return (np.nan, np.nan)

    data = np.array(data)
    n = len(data)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (lower, upper)


def required_n_for_power(effect_size: float, power: float = 0.8,
                          alpha: float = 0.05) -> int:
    """
    Calculate required sample size for given power.

    Uses approximation for two-sample t-test.
    """
    if effect_size <= 0:
        return float('inf')

    # Z-scores for power and alpha
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    # Sample size formula (per group)
    n = 2 * ((z_alpha + z_power) / effect_size) ** 2

    return int(np.ceil(n))


def power_at_n(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """
    Calculate power for given sample size and effect size.
    """
    if effect_size <= 0 or n <= 0:
        return 0.0

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ncp = effect_size * np.sqrt(n / 2)  # Non-centrality parameter

    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    return power


def compute_power_curve(effect_size: float, n_range: List[int],
                        alpha: float = 0.05) -> Dict:
    """
    Compute power at various sample sizes.
    """
    powers = [power_at_n(effect_size, n, alpha) for n in n_range]
    return {
        'n_values': n_range,
        'power_values': powers,
        'effect_size': effect_size,
        'alpha': alpha
    }


def run_effect_size_analysis(compression_results: Dict) -> Dict:
    """
    Run Phase 3 effect size analysis.

    Args:
        compression_results: Results from Phase 2

    Returns:
        Effect size analysis results
    """
    print("\nPhase 3: Effect Size Estimation")
    print("=" * 50)

    results = compression_results.get('results', {})

    analysis = {}

    # Effect sizes for each condition vs baseline/noise
    boost_deltas = results.get('boost', {}).get('all_deltas', [])
    noise_deltas = results.get('noise', {}).get('all_deltas', [])
    decay_deltas = results.get('decay', {}).get('all_deltas', [])
    baseline_deltas = results.get('baseline', {}).get('all_deltas', [])

    # Cohen's d: boost vs noise
    if boost_deltas and noise_deltas:
        d_boost_noise = cohens_d(boost_deltas, noise_deltas)
        print(f"  Cohen's d (boost vs noise): {d_boost_noise:.3f}")
        analysis['d_boost_vs_noise'] = d_boost_noise
    else:
        analysis['d_boost_vs_noise'] = None

    # Cohen's d: decay vs noise
    if decay_deltas and noise_deltas:
        d_decay_noise = cohens_d(decay_deltas, noise_deltas)
        print(f"  Cohen's d (decay vs noise): {d_decay_noise:.3f}")
        analysis['d_decay_vs_noise'] = d_decay_noise
    else:
        analysis['d_decay_vs_noise'] = None

    # One-sample d for boost (vs delta=0)
    if boost_deltas:
        d_boost = cohens_d_one_sample(boost_deltas, 0.0)
        print(f"  Cohen's d (boost vs 0): {d_boost:.3f}")
        analysis['d_boost'] = d_boost
    else:
        analysis['d_boost'] = None

    # Use largest effect size for power calculation
    effect_sizes = [v for v in [analysis.get('d_boost_vs_noise'),
                                 analysis.get('d_decay_vs_noise'),
                                 analysis.get('d_boost')] if v is not None]

    if effect_sizes:
        main_effect = max(effect_sizes)
        analysis['main_effect_size'] = main_effect

        # Required N for power = 0.8
        n_required = required_n_for_power(main_effect, power=0.8)
        print(f"\n  Main effect size: {main_effect:.3f}")
        print(f"  Required N (power=0.8): {n_required}")
        analysis['n_required_power80'] = n_required

        # Power curve
        n_range = [10, 20, 30, 50, 75, 100, 150, 200]
        power_curve = compute_power_curve(main_effect, n_range)
        analysis['power_curve'] = power_curve
    else:
        analysis['main_effect_size'] = None
        analysis['n_required_power80'] = None
        analysis['power_curve'] = None

    # Confidence intervals for each condition
    analysis['confidence_intervals'] = {}
    for condition, data in results.items():
        deltas = data.get('all_deltas', [])
        if deltas:
            ci = bootstrap_ci(deltas)
            analysis['confidence_intervals'][condition] = {
                'mean': np.mean(deltas),
                'ci_95': ci
            }
            print(f"  {condition}: mean={np.mean(deltas):.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    # Evaluate predictions
    predictions = {}

    # P3.1: Effect size > 0.3
    main_d = analysis.get('main_effect_size', 0)
    predictions['P3.1'] = {
        'description': 'Effect size measurable',
        'threshold': "Cohen's d > 0.3",
        'measured': main_d,
        'passed': main_d is not None and main_d > 0.3
    }

    # P3.2: Required N < 100
    n_req = analysis.get('n_required_power80', float('inf'))
    predictions['P3.2'] = {
        'description': 'Reasonable sample size',
        'threshold': 'N < 100 for power=0.8',
        'measured': n_req,
        'passed': n_req is not None and n_req < 100
    }

    print("\nPredictions:")
    for pred_id, pred in predictions.items():
        status = 'PASS' if pred['passed'] else 'FAIL'
        print(f"  {pred_id}: {pred['description']} - {status}")

    analysis['predictions'] = predictions

    return analysis
