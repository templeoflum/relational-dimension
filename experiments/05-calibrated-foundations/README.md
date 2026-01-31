# Experiment 05: Calibrated Foundations

## Overview

This experiment implements **protocol improvements** from the Phase 7-8 Red Team/Meta-Analysis before testing new hypotheses. It uses a **calibration-first** approach: the measurement method must demonstrably recover known ground truth before we trust compression measurements.

## Key Protocol Changes

| Previous Issue | This Experiment's Solution |
|----------------|----------------------------|
| Baseline δ ≠ 0 | **Calibration Gate:** Must achieve δ < 0.05 on known systems |
| Threshold adjustment | **Pre-registered & LOCKED** - no changes allowed |
| High variance on random graphs | **Regular lattices only** in Phase 1 |
| No power analysis | **Power analysis** determines sample sizes |
| Missing diagnostics | **Mandatory outputs** - blocking if missing |
| Method disagreement | **Agreement gate:** exclude if diff > 0.5 |

## Three-Phase Structure

### Phase 1: Calibration (BLOCKING)

Test on systems with **known ground truth**:
- 1D chain (d = 1)
- 2D lattice (d = 2)
- 3D lattice (d = 3)

**Must pass ALL gates before proceeding:**
- C1-C3: δ < 0.05 for each system
- C4: Isomap/MDS agreement < 0.3
- C5: std(δ) < 0.03 (reproducibility)

### Phase 2: Compression Tests

If Phase 1 passes, test compression on:
- 2D lattice + global correlation boost (expect δ > 0.1)
- 2D lattice + random noise (expect δ ≈ 0)
- 2D lattice + distance-decay correlation (expect δ > 0.05)

### Phase 3: Effect Size

Characterize the effect:
- Cohen's d for compression signal
- Required N for power = 0.8
- 95% confidence intervals

## Quick Start

```bash
cd experiments/05-calibrated-foundations

# Run tests first
pytest src/tests/ -v

# Run experiment (will stop at Phase 1 if calibration fails)
python src/main.py
```

## Pre-Registered Predictions (LOCKED)

### Phase 1: Calibration

| ID | Description | Threshold |
|----|-------------|-----------|
| P1.1 | 2D lattice δ | \|δ\| < 0.05 |
| P1.2 | 3D lattice δ | \|δ\| < 0.05 |
| P1.3 | 1D chain δ | \|δ\| < 0.05 |
| P1.4 | Method agreement | diff < 0.3 |
| P1.5 | Reproducibility | std < 0.03 |

### Phase 2: Compression

| ID | Description | Threshold |
|----|-------------|-----------|
| P2.1 | Global boost compression | δ > 0.1 |
| P2.2 | Noise control | \|δ\| < 0.05 |
| P2.3 | Distance-decay compression | δ > 0.05 |
| P2.4 | Ordering: boost > decay > noise | true |

### Phase 3: Effect Size

| ID | Description | Threshold |
|----|-------------|-----------|
| P3.1 | Effect size | Cohen's d > 0.3 |
| P3.2 | Required N | N < 100 for power=0.8 |

## Mandatory Outputs

All outputs are **blocking** - experiment marked incomplete if missing.

### Phase 1
- `output/phase1/calibration_results.json`
- `reports/lattice_2d_embedding.png`
- `reports/lattice_3d_embedding.png`
- `reports/method_comparison.png`

### Phase 2
- `output/phase2/compression_results.json`
- `reports/compression_by_condition.png`
- `reports/effect_ordering.png`

### Phase 3
- `output/phase3/effect_size_analysis.json`
- `reports/power_curve.png`
- `reports/confidence_intervals.png`

## Success Criteria

| Outcome | Interpretation |
|---------|----------------|
| Phase 1 fails | Method fundamentally broken - major revision |
| Phase 1 passes, Phase 2 fails | No compression effect - consider archiving |
| All phases pass, d > 0.3 | Effect exists, proceed to Exp06 |

## Commitment

By executing this experiment, we commit to:
1. **NOT adjusting thresholds** - results stand as measured
2. **Stopping at Phase 1 failure** - no proceeding without calibration
3. **Reporting all failures honestly**
4. **Generating all mandatory outputs**

---

*Implements protocol improvements from Phase 8 Meta-Analysis*
