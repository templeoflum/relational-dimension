# Development Log: Experiment 03

## Session 1: Implementation and Execution

### Completed
- Created experiment directory structure
- Implemented `consistent_methods.py`:
  - `full_mds_dimension()` - Full MDS without approximations
  - `full_isomap_dimension()` - Full Isomap without approximations
  - `extract_dimension_consistent()` - Both methods with validation
  - `find_dimension_with_interpolation()` - Fractional dimension via interpolation
  - `compute_compression_ratio()` - Same as Exp01/02
- Implemented `scaling_robust.py`:
  - Updated thresholds for Exp03 (P1: 0.85, P2: 0.80, P3: 0.15, P4: 0.30, P5: 0.0, P6: 0.05)
  - `fit_log_model()` - Same as Exp02
  - `fit_sqrt_model()` - New alternative model
  - P1-P6 evaluators with new thresholds
- Implemented `main.py`:
  - Test matrix runner (120 configurations)
  - Model fitting pipeline
  - Visualization generation (4 figures)
- Created unit tests (55 tests, all passing)
- Ran experiment successfully
- Created paper (LaTeX + PDF + README.md)

### Key Results

**Predictions: 3/6 passed**

| ID | Description | Threshold | Measured | Result |
|----|-------------|-----------|----------|--------|
| P1 | Monotonic Scaling | r > 0.85 | 0.64 | FAIL |
| P2 | Logarithmic Scaling | R² > 0.80 | 0.21 | FAIL |
| P3 | Method Consistency | diff < 0.15 | 0.12 | PASS |
| P4 | Predictive Validity | error < 30% | 14.6% | PASS |
| P5 | Non-Saturation | delta > 0 | 0.016 | PASS |
| P6 | Variance Reduction | SE < 0.05 | 0.088 | FAIL |

### Key Findings

1. **Methodological fix succeeded**: No sign reversal between N=500 and N=750
   - Exp02: delta(500) = 0.39, delta(750) = -0.15 (sign reversal!)
   - Exp03: delta(500) = 0.45, delta(750) = 0.33 (no reversal)

2. **Predictive validity restored**: 14.6% error vs 594% in Exp02

3. **High variance persists**: Standard deviations 0.28-0.62 across graph sizes
   - Anomalous dip at N=300 (delta = 0.185)
   - Some individual replications show negative delta

4. **Neither log nor sqrt model fits well**: R² = 0.21 (log), R² = 0.26 (sqrt)

### Comparison with Previous Experiments

| Metric | Exp01 | Exp02 | Exp03 |
|--------|-------|-------|-------|
| Predictions passed | 1/5 | 0/6 | 3/6 |
| delta(N=1000) | -- | -0.10 | 0.46 |
| Sign reversal? | -- | Yes | No |
| Extrapolation error | -- | 594% | 14.6% |

### Lessons Learned

1. Full methods are essential - sparse approximations introduce artifacts larger than the effect
2. High variance is inherent to the dimension estimation approach, not the sparse methods
3. The compression effect is real but noisy - may need different methodology

### Potential Next Steps

1. Try alternative dimension estimators (persistent homology, local PCA)
2. Use fixed graph topology with varying correlations to isolate effect
3. Increase replications further (50+ per N)
4. Test different correlation patterns
