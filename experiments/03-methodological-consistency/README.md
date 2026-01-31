# Experiment 03: Methodological Consistency and Robust Scaling

## Overview

This experiment addresses the methodological failure in Experiment 02: sparse methods (Landmark MDS, Sparse Isomap) introduced systematic bias that caused sign reversal in compression measurements. We use **consistent full methods** across all graph sizes (N = 50 to N = 1000) to test whether the scaling law holds.

## Research Question

Does compression scaling follow a logarithmic law when measured with consistent methodology across all graph sizes?

## Quick Start

```bash
# Run experiment
cd experiments/03-methodological-consistency
python src/main.py

# Run tests
pytest src/tests/ -v
```

## Predictions

| ID | Description | Threshold | Status | Measured |
|----|-------------|-----------|--------|----------|
| P1 | Monotonic Scaling | Spearman r > 0.85 | FAIL | 0.64 |
| P2 | Logarithmic Form | R² > 0.80 | FAIL | 0.21 |
| P3 | Method Consistency | \|δ(750) - δ(500)\| < 0.15 | PASS | 0.12 |
| P4 | Predictive Validity | Error < 30% | PASS | 14.6% |
| P5 | Non-Saturation | δ(1000) > δ(500) | PASS | 0.016 |
| P6 | Variance Reduction | SE < 0.05 | FAIL | 0.088 |

**Result: 3/6 predictions passed. Methodological fix succeeded (P3, P4), but high variance persists.**

## Key Methodological Changes from Exp02

1. **Full methods only** - No Landmark MDS or Sparse Isomap
2. **Increased replications** - 15-20 per N (vs 5-10 in Exp02)
3. **Method validation** - MDS and Isomap must agree within 0.5
4. **Continuity check** - Explicit verification of no discontinuity at N=500/750 boundary

## Test Matrix

| N | Replications | Purpose |
|---|--------------|---------|
| 50 | 20 | Anchor, reduce variance |
| 100 | 20 | Anchor, reduce variance |
| 200 | 20 | Anchor, reduce variance |
| 300 | 15 | New data point |
| 500 | 15 | Training set boundary |
| 750 | 15 | Test set (was sparse in Exp02) |
| 1000 | 15 | Extrapolation test |

**Total:** 120 configurations

## Output Files

- `output/metrics.json` - Full experiment results
- `reports/scaling_curve.png` - δ vs log(N) with 95% CI
- `reports/method_comparison.png` - MDS vs Isomap agreement
- `reports/residuals.png` - Model residuals
- `reports/predictions_summary.png` - P1-P6 results
- `paper/main.pdf` - Compiled paper

## Dependencies

- Experiment 01 modules: `graph_generation.py`, `correlation.py`, `dimension.py`, `metrics.py`
- Experiment 02 modules: `scaling_analysis.py` (modified)
- numpy, scipy, scikit-learn, matplotlib, tqdm
