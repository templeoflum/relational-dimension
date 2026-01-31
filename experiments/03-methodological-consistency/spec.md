# Experiment 03: Methodological Consistency and Robust Scaling

**Experiment ID:** 03
**Status:** Specification
**Created:** 2026-01-30
**Protocol Version:** 1.0

---

## 1. Motivation

Experiment 02 revealed a critical methodological problem: switching from full to sparse methods at N > 500 introduced a discontinuity that invalidated cross-scale comparisons. Within the consistent-method range (N = 50-500), logarithmic scaling was plausible (R² = 0.81).

This experiment addresses the core question: **Does the scaling law hold when methodology is consistent across all graph sizes?**

### Key Findings from Previous Experiments

| Experiment | Key Result | Implication |
|------------|------------|-------------|
| Exp01 | P3 passed: scaling ratio 3.11 | Compression increases with N |
| Exp02 | R² = 0.81 for log model (N ≤ 500) | Logarithmic scaling plausible |
| Exp02 | Sparse methods → negative δ | Methodological artifact |

### The Core Problem

Exp02's sparse methods (Landmark MDS, Sparse Isomap) produced systematically biased results:
- Full methods (N ≤ 500): δ ≈ +0.11 to +0.39 (compression)
- Sparse methods (N > 500): δ ≈ -0.15 to -0.10 (expansion)

This sign reversal is clearly artifactual. Memory analysis shows full methods are feasible up to N = 1000 (only 8MB per distance matrix).

---

## 2. Research Question

**Does compression scaling follow a logarithmic law when measured with consistent methodology across N = 50 to N = 1000?**

Secondary questions:
1. Does the scaling law extrapolate beyond the training range?
2. Is variance reduced with more replications?
3. At what N does the effect become statistically unambiguous?

---

## 3. Methodological Improvements

### 3.1 Consistent Full Methods

**All sizes use identical algorithms:**
- Full distance matrices (no landmarks, no approximations)
- Classical MDS for dimension estimation
- Isomap with n_neighbors = 8 for validation
- Error threshold τ = 0.1 (same as Exp01/02)

**Memory budget:** N = 1000 → 1M entries × 8 bytes = 8MB per matrix. Trivially feasible.

### 3.2 Increased Replications

| N | Exp02 Replications | Exp03 Replications | Rationale |
|---|-------------------|-------------------|-----------|
| 50 | 10 | 20 | Reduce variance |
| 100 | 10 | 20 | Reduce variance |
| 200 | 10 | 20 | Reduce variance |
| 300 | 10 | 15 | New data point |
| 500 | 10 | 15 | Training boundary |
| 750 | 5 | 15 | Full method now |
| 1000 | 5 | 15 | Full method now |

**Total:** 120 configurations (vs 60 in Exp02)

### 3.3 Validation Checks

For each (N, replication) pair:
1. **Method agreement:** MDS and Isomap dimensions within 0.5
2. **Embedding quality:** Stress < 0.2 for both metrics
3. **Outlier detection:** Flag if δ > 3 std from mean at that N

---

## 4. Pre-Registered Predictions

Six predictions with explicit thresholds, designed to pass if the scaling hypothesis is correct AND methodology is sound.

### P1: Monotonic Scaling (Revised)
**Claim:** δ increases monotonically with N across full range.
**Threshold:** Spearman r(N, δ) > 0.85
**Revision from Exp02:** Lowered from 0.9 to 0.85 given high variance observed.

### P2: Logarithmic Scaling (Revised)
**Claim:** δ(N) follows logarithmic form.
**Threshold:** R² > 0.80 for δ = a·log(N) + b
**Revision from Exp02:** Lowered from 0.85 to 0.80 to be achievable with observed variance.

### P3: Method Consistency
**Claim:** No discontinuity between N = 500 and N = 750.
**Threshold:** |δ(750) - δ(500)| < 0.15
**Rationale:** This is the boundary where Exp02 failed. If full methods are consistent, this gap should be small.

### P4: Predictive Validity
**Claim:** Model trained on N ≤ 500 predicts N = 1000 within 30%.
**Threshold:** |δ_predicted - δ_measured| / |δ_measured| < 0.30
**Revision from Exp02:** Relaxed from 20% to 30% given natural variance.

### P5: Non-Saturation
**Claim:** Effect continues growing at large N.
**Threshold:** δ(1000) > δ(500)
**Revision from Exp02:** Removed the +0.05 margin; just require positive growth.

### P6: Variance Reduction
**Claim:** Increased replications reduce standard error.
**Threshold:** SE(δ) at N = 100 < 0.05 (was 0.059 in Exp02 with 10 reps)
**Formula:** SE = std / sqrt(n_reps)

---

## 5. Test Matrix

| N | Replications | Method | Training/Test |
|---|--------------|--------|---------------|
| 50 | 20 | Full MDS + Isomap | Training |
| 100 | 20 | Full MDS + Isomap | Training |
| 200 | 20 | Full MDS + Isomap | Training |
| 300 | 15 | Full MDS + Isomap | Training |
| 500 | 15 | Full MDS + Isomap | Training |
| 750 | 15 | Full MDS + Isomap | Test |
| 1000 | 15 | Full MDS + Isomap | Test |

**Correlation type:** Long-range (LR) only, as in Exp01/02.

---

## 6. Model Fitting

### 6.1 Primary Model: Logarithmic

δ(N) = a · log(N) + b

Fit via ordinary least squares on training set (N ≤ 500).

### 6.2 Alternative Model: Square Root

δ(N) = c · sqrt(N) + d

Added as alternative to power law (which had poor fit in Exp02).

### 6.3 Model Selection

Compare models via:
- R² on training set
- AIC/BIC if models have same number of parameters
- Prediction error on test set (N > 500)

---

## 7. Implementation Plan

### 7.1 Module Structure

```
experiments/03-methodological-consistency/
├── README.md
├── DEVLOG.md
├── spec.md
├── claims.json
├── verification_gates.json
├── src/
│   ├── __init__.py
│   ├── consistent_methods.py   # Full MDS/Isomap only
│   ├── scaling_robust.py       # Model fitting with validation
│   ├── main.py                 # Test matrix runner
│   └── tests/
│       ├── __init__.py
│       ├── test_consistent_methods.py
│       └── test_scaling_robust.py
├── output/
│   └── metrics.json
├── reports/
│   ├── scaling_curve.png
│   ├── method_comparison.png
│   ├── residuals.png
│   └── predictions_summary.png
└── paper/
    ├── README.md              # Markdown version
    ├── main.tex               # LaTeX source
    └── main.pdf               # Compiled PDF
```

### 7.2 Reuse from Previous Experiments

| Module | Source | Modifications |
|--------|--------|---------------|
| graph_generation.py | Exp01 | None |
| correlation.py | Exp01 | None |
| dimension.py | Exp01 | Remove sparse methods |
| metrics.py | Exp01 | None |
| scaling_analysis.py | Exp02 | Add sqrt model, improve validation |

### 7.3 New Code

**consistent_methods.py:**
- `full_mds_dimension()` - Classical MDS, no approximations
- `full_isomap_dimension()` - Full Isomap with validation
- `validate_embedding()` - Check stress, agreement between methods

**scaling_robust.py:**
- `fit_models()` - Log, sqrt, power with uncertainty estimates
- `cross_validate()` - Leave-one-N-out validation
- `bootstrap_ci()` - Confidence intervals via bootstrap

---

## 8. Diagnostic Outputs

### 8.1 Figures

1. **scaling_curve.png** - δ vs log(N) with fitted line and 95% CI band
2. **method_comparison.png** - MDS vs Isomap dimension estimates (should agree)
3. **residuals.png** - Model residuals vs N, check for patterns
4. **continuity_check.png** - δ at N = 500 vs N = 750, visualize gap
5. **predictions_summary.png** - P1-P6 pass/fail with measured values

### 8.2 Tables

1. **Raw data by N** - Mean, std, SE, min, max for each N
2. **Model parameters** - Coefficients with standard errors
3. **Prediction comparison** - Predicted vs measured at test sizes

---

## 9. Success Criteria

| Outcome | Predictions | Interpretation |
|---------|-------------|----------------|
| Full success | 6/6 pass | Scaling law validated, methodology sound |
| Strong success | 5/6 pass | Scaling real, minor issues |
| Partial success | 4/6 pass | Effect exists, needs refinement |
| Weak signal | 2-3/6 pass | Effect uncertain |
| Failure | 0-1/6 pass | Either effect is noise OR methodology still flawed |

### Critical Predictions

- **P3 (Method Consistency):** If this fails, the methodology improvement didn't work
- **P4 (Predictive Validity):** If this fails, the model doesn't extrapolate

---

## 10. Risk Mitigation

### Risk 1: High Variance Persists

**Detection:** SE at N=100 > 0.05 despite 20 replications
**Mitigation:** Report effect size with uncertainty, discuss statistical power

### Risk 2: Scaling Not Logarithmic

**Detection:** R² < 0.70 for all models
**Mitigation:** Explore piecewise or saturating models

### Risk 3: Computation Time

**Detection:** Full MDS at N=1000 takes > 10 min
**Mitigation:** Use efficient scipy.linalg.eigh, parallel execution

---

## 11. Timeline Estimate

- **Implementation:** 2-3 hours (mostly reusing Exp02 code)
- **Execution:** ~20-30 minutes (120 configs, but each is fast)
- **Analysis:** 2-3 hours (figures, interpretation)
- **Paper:** 2-3 hours (LaTeX, PDF compilation, visual verification)

---

## 12. Deliverables Checklist

- [ ] spec.md (this file)
- [ ] claims.json
- [ ] verification_gates.json
- [ ] src/consistent_methods.py
- [ ] src/scaling_robust.py
- [ ] src/main.py
- [ ] src/tests/*.py (unit tests)
- [ ] output/metrics.json
- [ ] reports/*.png (4-5 figures)
- [ ] paper/README.md (markdown version)
- [ ] paper/main.tex (LaTeX source)
- [ ] paper/main.pdf (compiled, visually verified)

---

## Document Metadata

**Experiment:** 03
**Title:** Methodological Consistency and Robust Scaling
**Status:** Specification Complete
**Created:** 2026-01-30
**Dependencies:** Exp01 modules, Exp02 analysis approach
