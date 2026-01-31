"""
Tests for scaling_robust.py
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scaling_robust import (
    THRESHOLDS,
    fit_log_model,
    fit_sqrt_model,
    predict_delta,
    compare_models,
    evaluate_p1_monotonicity,
    evaluate_p2_log_scaling,
    evaluate_p3_method_consistency,
    evaluate_p4_predictive_validity,
    evaluate_p5_non_saturation,
    evaluate_p6_variance_reduction,
    evaluate_all_predictions,
    summarize_predictions
)


class TestThresholds:
    """Verify thresholds are correctly set for Exp03."""

    def test_p1_threshold(self):
        assert THRESHOLDS['P1_spearman'] == 0.85

    def test_p2_threshold(self):
        assert THRESHOLDS['P2_r2'] == 0.80

    def test_p3_threshold(self):
        assert THRESHOLDS['P3_continuity'] == 0.15

    def test_p4_threshold(self):
        assert THRESHOLDS['P4_relative_error'] == 0.30

    def test_p5_threshold(self):
        assert THRESHOLDS['P5_growth'] == 0.0

    def test_p6_threshold(self):
        assert THRESHOLDS['P6_se'] == 0.05


class TestFitLogModel:
    """Tests for fit_log_model."""

    def test_perfect_log_relationship(self):
        """Perfect log data should give R^2 = 1."""
        N = np.array([50, 100, 200, 400])
        delta = 0.1 * np.log(N) + 0.5

        result = fit_log_model(N, delta)

        assert result['valid'] is True
        assert result['r2'] > 0.99
        assert abs(result['a'] - 0.1) < 0.01
        assert abs(result['b'] - 0.5) < 0.01

    def test_noisy_log_relationship(self):
        """Noisy log data should have R^2 < 1."""
        np.random.seed(42)
        N = np.array([50, 100, 200, 400])
        delta = 0.1 * np.log(N) + 0.5 + np.random.normal(0, 0.1, 4)

        result = fit_log_model(N, delta)

        assert result['valid'] is True
        assert 0 < result['r2'] < 1

    def test_insufficient_data(self):
        """Single point should be invalid."""
        N = np.array([100])
        delta = np.array([0.3])

        result = fit_log_model(N, delta)

        assert result['valid'] is False

    def test_returns_formula(self):
        """Should return formula string."""
        N = np.array([50, 100, 200])
        delta = np.array([0.1, 0.2, 0.3])

        result = fit_log_model(N, delta)

        assert 'formula' in result
        assert 'log(N)' in result['formula']


class TestFitSqrtModel:
    """Tests for fit_sqrt_model."""

    def test_perfect_sqrt_relationship(self):
        """Perfect sqrt data should give R^2 = 1."""
        N = np.array([25, 100, 225, 400])
        delta = 0.02 * np.sqrt(N) + 0.1

        result = fit_sqrt_model(N, delta)

        assert result['valid'] is True
        assert result['r2'] > 0.99

    def test_returns_coefficients(self):
        """Should return c and d coefficients."""
        N = np.array([50, 100, 200])
        delta = np.array([0.1, 0.2, 0.3])

        result = fit_sqrt_model(N, delta)

        assert 'c' in result
        assert 'd' in result


class TestPredictDelta:
    """Tests for predict_delta."""

    def test_log_model_prediction(self):
        """Should predict using log model."""
        model = {'a': 0.1, 'b': -0.2, 'valid': True, 'model': 'log'}

        pred = predict_delta(model, 100)

        expected = 0.1 * np.log(100) - 0.2
        assert abs(pred - expected) < 0.001

    def test_sqrt_model_prediction(self):
        """Should predict using sqrt model."""
        model = {'c': 0.02, 'd': 0.1, 'valid': True, 'model': 'sqrt'}

        pred = predict_delta(model, 100)

        expected = 0.02 * np.sqrt(100) + 0.1
        assert abs(pred - expected) < 0.001

    def test_invalid_model(self):
        """Invalid model should return 0."""
        model = {'a': 0.1, 'b': 0.2, 'valid': False, 'model': 'log'}

        pred = predict_delta(model, 100)

        assert pred == 0.0


class TestCompareModels:
    """Tests for compare_models."""

    def test_log_wins(self):
        """Log model with higher R^2 should win."""
        log_fit = {'r2': 0.9, 'valid': True}
        sqrt_fit = {'r2': 0.7, 'valid': True}

        result = compare_models(log_fit, sqrt_fit)

        assert result['winner'] == 'log'
        assert result['winner_r2'] == 0.9

    def test_sqrt_wins(self):
        """Sqrt model with higher R^2 should win."""
        log_fit = {'r2': 0.6, 'valid': True}
        sqrt_fit = {'r2': 0.85, 'valid': True}

        result = compare_models(log_fit, sqrt_fit)

        assert result['winner'] == 'sqrt'
        assert result['winner_r2'] == 0.85


class TestEvaluateP1:
    """Tests for evaluate_p1_monotonicity."""

    def test_perfect_monotonic(self):
        """Perfectly monotonic should pass."""
        N = np.array([50, 100, 200, 500, 1000])
        delta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = evaluate_p1_monotonicity(N, delta)

        assert result['passed'] is True
        assert result['measured'] > 0.99  # Spearman r for perfect monotonic

    def test_non_monotonic(self):
        """Non-monotonic should fail."""
        N = np.array([50, 100, 200, 500, 1000])
        delta = np.array([0.1, 0.3, 0.2, 0.4, 0.35])

        result = evaluate_p1_monotonicity(N, delta)

        assert result['measured'] < 0.85

    def test_insufficient_data(self):
        """Too few points should fail."""
        N = np.array([50, 100])
        delta = np.array([0.1, 0.2])

        result = evaluate_p1_monotonicity(N, delta)

        assert result['passed'] is False


class TestEvaluateP2:
    """Tests for evaluate_p2_log_scaling."""

    def test_good_fit_passes(self):
        """R^2 > 0.80 should pass."""
        log_model = {'r2': 0.85, 'valid': True, 'formula': 'test'}

        result = evaluate_p2_log_scaling(log_model)

        assert result['passed'] is True
        assert result['threshold'] == 0.80

    def test_poor_fit_fails(self):
        """R^2 < 0.80 should fail."""
        log_model = {'r2': 0.75, 'valid': True, 'formula': 'test'}

        result = evaluate_p2_log_scaling(log_model)

        assert result['passed'] is False


class TestEvaluateP3:
    """Tests for evaluate_p3_method_consistency."""

    def test_small_gap_passes(self):
        """Small gap between N=500 and N=750 should pass."""
        result = evaluate_p3_method_consistency(delta_500=0.35, delta_750=0.38)

        assert result['passed'] is True
        assert result['measured'] < 0.15

    def test_large_gap_fails(self):
        """Large gap should fail."""
        result = evaluate_p3_method_consistency(delta_500=0.35, delta_750=0.55)

        assert result['passed'] is False

    def test_sign_change_fails(self):
        """Sign change should produce large gap."""
        result = evaluate_p3_method_consistency(delta_500=0.35, delta_750=-0.10)

        assert result['passed'] is False
        assert result['measured'] > 0.15


class TestEvaluateP4:
    """Tests for evaluate_p4_predictive_validity."""

    def test_good_prediction_passes(self):
        """Prediction within 30% should pass."""
        result = evaluate_p4_predictive_validity(delta_predicted=0.45, delta_measured=0.50)

        assert result['passed'] is True
        assert result['measured'] < 0.30

    def test_poor_prediction_fails(self):
        """Prediction off by >30% should fail."""
        result = evaluate_p4_predictive_validity(delta_predicted=0.45, delta_measured=0.30)

        assert result['passed'] is False

    def test_zero_measured_handles_gracefully(self):
        """Zero measured value should not crash."""
        result = evaluate_p4_predictive_validity(delta_predicted=0.1, delta_measured=0.0)

        assert 'passed' in result


class TestEvaluateP5:
    """Tests for evaluate_p5_non_saturation."""

    def test_growth_passes(self):
        """delta(1000) > delta(500) should pass."""
        result = evaluate_p5_non_saturation(delta_500=0.35, delta_1000=0.45)

        assert result['passed'] is True

    def test_saturation_fails(self):
        """delta(1000) <= delta(500) should fail."""
        result = evaluate_p5_non_saturation(delta_500=0.35, delta_1000=0.30)

        assert result['passed'] is False


class TestEvaluateP6:
    """Tests for evaluate_p6_variance_reduction."""

    def test_low_variance_passes(self):
        """SE < 0.05 should pass."""
        result = evaluate_p6_variance_reduction(se_delta_n100=0.04)

        assert result['passed'] is True

    def test_high_variance_fails(self):
        """SE >= 0.05 should fail."""
        result = evaluate_p6_variance_reduction(se_delta_n100=0.06)

        assert result['passed'] is False


class TestSummarize:
    """Tests for summarize_predictions."""

    def test_full_success(self):
        """All passing should be full_success."""
        predictions = {
            'P1': {'passed': True},
            'P2': {'passed': True},
            'P3': {'passed': True},
            'P4': {'passed': True},
            'P5': {'passed': True},
            'P6': {'passed': True},
        }

        summary = summarize_predictions(predictions)

        assert summary['status'] == 'full_success'
        assert summary['predictions_passed'] == 6

    def test_partial_success(self):
        """4 passing should be partial_success."""
        predictions = {
            'P1': {'passed': True},
            'P2': {'passed': True},
            'P3': {'passed': True},
            'P4': {'passed': True},
            'P5': {'passed': False},
            'P6': {'passed': False},
        }

        summary = summarize_predictions(predictions)

        assert summary['status'] == 'partial_success'

    def test_failure(self):
        """0-1 passing should be failure."""
        predictions = {
            'P1': {'passed': False},
            'P2': {'passed': True},
            'P3': {'passed': False},
            'P4': {'passed': False},
            'P5': {'passed': False},
            'P6': {'passed': False},
        }

        summary = summarize_predictions(predictions)

        assert summary['status'] == 'failure'
