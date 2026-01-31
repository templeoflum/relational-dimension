"""
Experiment 02: Compression Scaling Laws

Investigates the functional form of compression scaling with graph size.
"""

from .dimension_improved import (
    continuous_dimension,
    landmark_mds,
    sparse_isomap,
    explained_variance_curve,
)

from .scaling_analysis import (
    fit_log_model,
    fit_power_model,
    predict_delta,
    compare_models,
)

__all__ = [
    'continuous_dimension',
    'landmark_mds',
    'sparse_isomap',
    'explained_variance_curve',
    'fit_log_model',
    'fit_power_model',
    'predict_delta',
    'compare_models',
]
