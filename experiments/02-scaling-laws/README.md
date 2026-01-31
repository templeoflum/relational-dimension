# Experiment 02: Compression Scaling Laws

## Overview

This experiment investigates the functional form of compression scaling with graph size, building on Experiment 01's finding (P3 passed: scaling ratio 3.11).

## Research Question

What is the functional form of compression scaling, and does it predict behavior at unseen graph sizes?

## Quick Start

```bash
# Run experiment
cd experiments/02-scaling-laws
python src/main.py

# Run tests
pytest src/tests/ -v
```

## Predictions

| ID | Description | Threshold | Status | Measured |
|----|-------------|-----------|--------|----------|
| P1 | Monotonic Scaling | Spearman r > 0.9 | FAIL | -0.29 |
| P2 | Logarithmic Form | R² > 0.85 | FAIL | 0.81 |
| P3 | Power Law Form | R² > 0.85, α > 0 | FAIL | 0.63 |
| P4 | Non-Saturation | δ(1000) > δ(500) + 0.05 | FAIL | -0.48 |
| P5 | Predictive Validity | Error < 20% | FAIL | 594% |
| P6 | Variance Reduction | Std < 0.15 | FAIL | 0.19 |

**Note:** Results affected by methodological transition at N>500 (sparse vs full methods).

## Test Matrix

| N | Replications |
|---|--------------|
| 50 | 10 |
| 100 | 10 |
| 200 | 10 |
| 300 | 10 |
| 500 | 10 |
| 750 | 5 |
| 1000 | 5 |

## Output Files

- `output/metrics.json` - Full experiment results
- `reports/scaling_curve.png` - δ vs N with fitted models
- `reports/residuals.png` - Model residuals
- `reports/prediction_test.png` - Predictive validity visualization
- `reports/predictions_summary.png` - P1-P6 results

## Dependencies

- Experiment 01 modules: `graph_generation.py`, `correlation.py`, `dimension.py`, `metrics.py`
- numpy, scipy, scikit-learn, matplotlib, tqdm
