# Experiment 02: Compression Scaling Laws

## Overview

Building on Experiment 01's robust finding (P3 passed: scaling ratio 3.11), this experiment investigates the **scaling law** governing how compression increases with graph size.

## Research Question

**What is the functional form of compression scaling, and does it predict behavior at unseen graph sizes?**

## Key Findings from Experiment 01

| N | δ (RGG-LR) | δ (RGG-NN) |
|---|------------|------------|
| 50 | 0.108 | 0.070 |
| 100 | 0.296 | 0.233 |
| 200 | 0.336 | 0.410 |
| **Ratio (200/50)** | **3.11** | **5.86** |

## Predictions (Pre-registered)

### P1: Monotonic Scaling
- **Claim:** δ increases monotonically with N for LR correlations
- **Threshold:** Spearman correlation r(N, δ) > 0.9

### P2: Functional Form - Logarithmic
- **Claim:** δ(N) follows δ = a·log(N) + b
- **Threshold:** R² > 0.85 for log fit

### P3: Functional Form - Power Law (Alternative)
- **Claim:** δ(N) follows δ = c·N^α
- **Threshold:** R² > 0.85 for power fit, α > 0

### P4: Saturation Test
- **Claim:** Compression does NOT saturate (no plateau)
- **Threshold:** δ(1000) > δ(500) + 0.05

### P5: Predictive Validity
- **Claim:** Model fitted on N ≤ 500 predicts δ(1000) within 20%
- **Threshold:** |δ_predicted - δ_measured| / δ_measured < 0.2

### P6: Improved Baseline
- **Claim:** With continuous dimension estimation, baseline variance drops
- **Threshold:** Std(δ) at N=100 < 0.15 (was 0.27 in Exp01)

## Test Matrix

| N | Replications | Purpose |
|---|--------------|---------|
| 50 | 10 | Anchor to Exp01 |
| 100 | 10 | Anchor to Exp01 |
| 200 | 10 | Anchor to Exp01 |
| 300 | 10 | New data point |
| 500 | 10 | New data point |
| 750 | 5 | New data point |
| 1000 | 5 | Extrapolation test |

**Total:** 70 configurations
**Correlation type:** LR only

## Methodological Improvements

1. **Continuous Dimension Estimation:** Explained variance ratio instead of threshold
2. **Sparse Distance Computation:** Landmark MDS for N > 500
3. **Bootstrap Confidence Intervals:** More efficient CI estimation
