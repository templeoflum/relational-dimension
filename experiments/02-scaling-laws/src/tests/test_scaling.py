"""
Tests for scaling analysis module.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scaling_analysis import (
    fit_log_model,
    fit_power_model,
    predict_delta,
    compare_models,
    evaluate_monotonicity,
    evaluate_log_fit,
    evaluate_power_fit,
    evaluate_saturation,
    evaluate_predictive_validity,
    evaluate_variance_reduction,
    evaluate_all_predictions,
    summarize_predictions
)


class TestFitLogModel:
    """Tests for fit_log_model function."""

    def test_perfect_log_data(self):
        """Should fit perfectly to log data."""
        N = np.array([50, 100, 200, 500, 1000])
        # δ = 0.1 * log(N) - 0.2
        delta = 0.1 * np.log(N) - 0.2

        result = fit_log_model(N, delta)

        assert result['valid']
        assert abs(result['a'] - 0.1) < 0.001
        assert abs(result['b'] - (-0.2)) < 0.001
        assert result['r2'] > 0.999

    def test_noisy_log_data(self):
        """Should fit reasonably to noisy log data."""
        np.random.seed(42)
        N = np.array([50, 100, 200, 500, 1000])
        delta = 0.1 * np.log(N) - 0.2 + np.random.normal(0, 0.02, len(N))

        result = fit_log_model(N, delta)

        assert result['valid']
        assert result['r2'] > 0.9

    def test_returns_formula(self):
        """Should include formula string."""
        N = np.array([50, 100, 200])
        delta = np.array([0.1, 0.2, 0.3])

        result = fit_log_model(N, delta)

        assert 'formula' in result
        assert 'log(N)' in result['formula']

    def test_insufficient_data(self):
        """Should handle insufficient data."""
        N = np.array([100])
        delta = np.array([0.2])

        result = fit_log_model(N, delta)

        assert not result['valid']


class TestFitPowerModel:
    """Tests for fit_power_model function."""

    def test_perfect_power_data(self):
        """Should fit perfectly to power law data."""
        N = np.array([50, 100, 200, 500, 1000])
        # δ = 0.01 * N^0.5
        delta = 0.01 * (N ** 0.5)

        result = fit_power_model(N, delta)

        assert result['valid']
        assert abs(result['alpha'] - 0.5) < 0.01
        assert result['r2'] > 0.99

    def test_noisy_power_data(self):
        """Should fit reasonably to noisy power data."""
        np.random.seed(42)
        N = np.array([50, 100, 200, 500, 1000])
        delta = 0.01 * (N ** 0.5) + np.random.normal(0, 0.01, len(N))

        result = fit_power_model(N, delta)

        assert result['valid']
        assert result['r2'] > 0.9

    def test_handles_negative_delta(self):
        """Should handle negative delta values gracefully."""
        N = np.array([50, 100, 200])
        delta = np.array([-0.1, 0.1, 0.2])

        result = fit_power_model(N, delta)

        # Should still work on positive values
        assert 'valid' in result


class TestPredictDelta:
    """Tests for predict_delta function."""

    def test_log_prediction(self):
        """Should predict correctly for log model."""
        model = {
            'valid': True,
            'model': 'log',
            'a': 0.1,
            'b': -0.2
        }

        pred = predict_delta(model, 100)
        expected = 0.1 * np.log(100) - 0.2

        assert abs(pred - expected) < 0.001

    def test_power_prediction(self):
        """Should predict correctly for power model."""
        model = {
            'valid': True,
            'model': 'power',
            'c': 0.01,
            'alpha': 0.5
        }

        pred = predict_delta(model, 100)
        expected = 0.01 * (100 ** 0.5)

        assert abs(pred - expected) < 0.001

    def test_invalid_model(self):
        """Should return 0 for invalid model."""
        model = {'valid': False}

        pred = predict_delta(model, 100)

        assert pred == 0.0


class TestCompareModels:
    """Tests for compare_models function."""

    def test_log_wins(self):
        """Should correctly identify when log model wins."""
        log_fit = {'valid': True, 'r2': 0.95}
        power_fit = {'valid': True, 'r2': 0.85}

        result = compare_models(log_fit, power_fit)

        assert result['winner'] == 'log'
        assert result['log_r2'] == 0.95
        assert result['power_r2'] == 0.85

    def test_power_wins(self):
        """Should correctly identify when power model wins."""
        log_fit = {'valid': True, 'r2': 0.80}
        power_fit = {'valid': True, 'r2': 0.90}

        result = compare_models(log_fit, power_fit)

        assert result['winner'] == 'power'

    def test_tie(self):
        """Should handle tie."""
        log_fit = {'valid': True, 'r2': 0.85}
        power_fit = {'valid': True, 'r2': 0.85}

        result = compare_models(log_fit, power_fit)

        assert result['winner'] == 'tie'

    def test_invalid_models(self):
        """Should handle invalid models."""
        log_fit = {'valid': False}
        power_fit = {'valid': True, 'r2': 0.90}

        result = compare_models(log_fit, power_fit)

        assert result['winner'] == 'power'
        assert result['log_r2'] == 0.0


class TestEvaluateMonotonicity:
    """Tests for evaluate_monotonicity (P1)."""

    def test_perfect_monotonic(self):
        """Should pass for perfectly monotonic data."""
        N = np.array([50, 100, 200, 500, 1000])
        delta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = evaluate_monotonicity(N, delta)

        assert result['passed']
        assert result['measured'] > 0.99  # Perfect Spearman correlation (allow float precision)

    def test_non_monotonic(self):
        """Should fail for non-monotonic data."""
        N = np.array([50, 100, 200, 500, 1000])
        delta = np.array([0.1, 0.3, 0.2, 0.5, 0.4])

        result = evaluate_monotonicity(N, delta)

        assert not result['passed']
        assert result['measured'] < 0.9


class TestEvaluateLogFit:
    """Tests for evaluate_log_fit (P2)."""

    def test_good_fit_passes(self):
        """Should pass for good R²."""
        model = {'valid': True, 'r2': 0.90, 'formula': 'δ = 0.1 * log(N) + 0.0'}

        result = evaluate_log_fit(model)

        assert result['passed']
        assert result['measured'] == 0.90

    def test_poor_fit_fails(self):
        """Should fail for poor R²."""
        model = {'valid': True, 'r2': 0.70, 'formula': 'δ = 0.1 * log(N) + 0.0'}

        result = evaluate_log_fit(model)

        assert not result['passed']


class TestEvaluatePowerFit:
    """Tests for evaluate_power_fit (P3)."""

    def test_good_fit_positive_alpha_passes(self):
        """Should pass for good R² and positive alpha."""
        model = {'valid': True, 'r2': 0.90, 'alpha': 0.5, 'formula': 'δ = 0.01 * N^0.5'}

        result = evaluate_power_fit(model)

        assert result['passed']

    def test_negative_alpha_fails(self):
        """Should fail for negative alpha."""
        model = {'valid': True, 'r2': 0.90, 'alpha': -0.5, 'formula': 'δ = 0.01 * N^-0.5'}

        result = evaluate_power_fit(model)

        assert not result['passed']


class TestEvaluateSaturation:
    """Tests for evaluate_saturation (P4)."""

    def test_non_saturated_passes(self):
        """Should pass when compression continues to grow."""
        result = evaluate_saturation(delta_500=0.3, delta_1000=0.4)

        assert result['passed']
        assert abs(result['measured'] - 0.1) < 0.001

    def test_saturated_fails(self):
        """Should fail when compression plateaus."""
        result = evaluate_saturation(delta_500=0.3, delta_1000=0.32)

        assert not result['passed']
        assert abs(result['measured'] - 0.02) < 0.001


class TestEvaluatePredictiveValidity:
    """Tests for evaluate_predictive_validity (P5)."""

    def test_accurate_prediction_passes(self):
        """Should pass for accurate prediction."""
        result = evaluate_predictive_validity(delta_predicted=0.38, delta_measured=0.40)

        assert result['passed']
        assert abs(result['measured'] - 0.05) < 0.001  # 5% error

    def test_poor_prediction_fails(self):
        """Should fail for inaccurate prediction."""
        result = evaluate_predictive_validity(delta_predicted=0.30, delta_measured=0.40)

        assert not result['passed']
        assert abs(result['measured'] - 0.25) < 0.001  # 25% error


class TestEvaluateVarianceReduction:
    """Tests for evaluate_variance_reduction (P6)."""

    def test_low_variance_passes(self):
        """Should pass for low variance."""
        result = evaluate_variance_reduction(std_delta_n100=0.10)

        assert result['passed']

    def test_high_variance_fails(self):
        """Should fail for high variance."""
        result = evaluate_variance_reduction(std_delta_n100=0.20)

        assert not result['passed']


class TestEvaluateAllPredictions:
    """Tests for evaluate_all_predictions."""

    def test_complete_results(self):
        """Should evaluate all predictions when data is complete."""
        results = {
            'N_values': [50, 100, 200, 500, 1000],
            'delta_means': [0.1, 0.2, 0.3, 0.35, 0.40],
            'delta_by_n': {500: [0.35], 1000: [0.40]},
            'std_by_n': {100: 0.10},
            'log_model': {'valid': True, 'r2': 0.90, 'formula': 'test'},
            'power_model': {'valid': True, 'r2': 0.85, 'alpha': 0.3, 'formula': 'test'},
            'delta_predicted_1000': 0.38,
            'delta_measured_1000': 0.40
        }

        predictions = evaluate_all_predictions(results)

        assert 'P1' in predictions
        assert 'P2' in predictions
        assert 'P3' in predictions
        assert 'P4' in predictions
        assert 'P5' in predictions
        assert 'P6' in predictions


class TestSummarizePredictions:
    """Tests for summarize_predictions."""

    def test_all_pass(self):
        """Should report success when all pass."""
        predictions = {
            'P1': {'passed': True},
            'P2': {'passed': True},
            'P3': {'passed': True},
            'P4': {'passed': True},
            'P5': {'passed': True},
            'P6': {'passed': True}
        }

        summary = summarize_predictions(predictions)

        assert summary['status'] == 'success'
        assert summary['predictions_passed'] == 6

    def test_monotonic_fails(self):
        """Should report falsified when P1 fails."""
        predictions = {
            'P1': {'passed': False},
            'P2': {'passed': True},
            'P3': {'passed': True}
        }

        summary = summarize_predictions(predictions)

        assert summary['status'] == 'falsified'
        assert not summary['monotonic_scaling_passed']

    def test_functional_form_detection(self):
        """Should detect which functional form is validated."""
        # Log only
        predictions = {'P2': {'passed': True}, 'P3': {'passed': False}, 'P1': {'passed': True}}
        summary = summarize_predictions(predictions)
        assert summary['functional_form'] == 'logarithmic'

        # Power only
        predictions = {'P2': {'passed': False}, 'P3': {'passed': True}, 'P1': {'passed': True}}
        summary = summarize_predictions(predictions)
        assert summary['functional_form'] == 'power_law'

        # Both
        predictions = {'P2': {'passed': True}, 'P3': {'passed': True}, 'P1': {'passed': True}}
        summary = summarize_predictions(predictions)
        assert summary['functional_form'] == 'both'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
