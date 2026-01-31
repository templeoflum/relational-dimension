# Experiment 05: Calibrated Foundations

## Overview

This experiment addresses the methodological issues identified in the Phase 7-8 Red Team/Meta-Analysis before testing any new hypotheses. It implements a **calibration-first** approach with strict protocol controls.

**Philosophy:** We cannot trust compression measurements until the method demonstrably recovers known ground truth.

---

## Protocol Improvements Implemented

| Issue from Meta-Analysis | Solution in Exp05 |
|--------------------------|-------------------|
| Baseline δ ≠ 0 for matched systems | Calibration Gate: Must achieve δ < 0.05 |
| Threshold adjustment (p-hacking) | All thresholds LOCKED before execution |
| High variance on random graphs | Test on regular lattices only (Phase 1) |
| No power analysis | Power analysis determines sample sizes |
| Missing diagnostic outputs | Mandatory output checklist with blocking |
| Isomap/MDS disagreement | Method Agreement Gate: exclude if diff > 0.5 |

---

## Experiment Structure

### Phase 1: Calibration (BLOCKING)

Must pass ALL calibration gates before proceeding to Phase 2.

**Purpose:** Validate that the measurement method works on known systems.

**Test Systems:**
1. 2D square lattice (d_true = 2)
2. 3D cubic lattice (d_true = 3)
3. 1D chain (d_true = 1)

**For each system:**
- Generate topology-matched correlation matrix
- Extract d_topo and d_corr using both Isomap and MDS
- Compute δ = (d_topo - d_corr) / d_topo

**Calibration Gates (ALL must pass):**

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| C1 | δ_2D = 0.00 ± 0.05 | 2D lattice must recover d=2 |
| C2 | δ_3D = 0.00 ± 0.05 | 3D lattice must recover d=3 |
| C3 | δ_1D = 0.00 ± 0.05 | 1D chain must recover d=1 |
| C4 | Isomap/MDS agreement < 0.3 for all | Methods must converge |
| C5 | std(δ) < 0.03 across 20 replications | Must be reproducible |

**If ANY calibration gate fails:** STOP. Fix methodology before proceeding.

### Phase 2: Controlled Compression Test

Only executed if Phase 1 passes.

**Purpose:** Test compression on systems with KNOWN expected compression.

**Test Systems:**
1. **2D lattice + global correlation boost** (expected: δ > 0)
   - Add uniform correlation ε to all pairs
   - Should create compression (d_corr < d_topo)

2. **2D lattice + noise** (expected: δ ≈ 0)
   - Add random noise to correlation matrix
   - Should NOT create systematic compression

3. **2D lattice + distance-decay correlation** (expected: δ > 0)
   - C(i,j) = exp(-α × dist(i,j))
   - More realistic long-range correlation model

### Phase 3: Effect Size Estimation

**Purpose:** Characterize the effect for future experiments.

**Outputs:**
- Effect size (Cohen's d) for compression signal
- Required N for power = 0.8 at α = 0.05
- Confidence intervals for δ

---

## Pre-Registered Predictions

**LOCKED - These thresholds will NOT be adjusted regardless of results.**

### Phase 1 Predictions (Calibration)

| ID | Description | Threshold | Justification |
|----|-------------|-----------|---------------|
| P1.1 | 2D lattice calibration | \|δ\| < 0.05 | Ground truth d=2 must be recovered |
| P1.2 | 3D lattice calibration | \|δ\| < 0.05 | Ground truth d=3 must be recovered |
| P1.3 | 1D chain calibration | \|δ\| < 0.05 | Ground truth d=1 must be recovered |
| P1.4 | Method agreement | diff < 0.3 | Isomap and MDS must agree |
| P1.5 | Reproducibility | std < 0.03 | Low variance required |

### Phase 2 Predictions (Compression)

| ID | Description | Threshold | Justification |
|----|-------------|-----------|---------------|
| P2.1 | Global boost creates compression | δ > 0.1 | Uniform correlation should compress |
| P2.2 | Noise control shows no compression | \|δ\| < 0.05 | Random noise shouldn't compress |
| P2.3 | Distance-decay creates compression | δ > 0.05 | LR correlation should compress |
| P2.4 | Boost > Decay > Noise ordering | δ_boost > δ_decay > δ_noise | Effect ordering |

### Phase 3 Predictions (Effect Size)

| ID | Description | Threshold | Justification |
|----|-------------|-----------|---------------|
| P3.1 | Effect size measurable | Cohen's d > 0.3 | Small but real effect |
| P3.2 | Reasonable power achievable | N_required < 100 | Practical sample size |

---

## Power Analysis

Based on Exp01-04 observations:
- Effect size (δ): ~0.25
- Standard deviation: ~0.35
- Current SNR: ~0.7

**Target:** Power = 0.80, α = 0.05

**Required N per condition (estimated):**
```
For Cohen's d = 0.7 (optimistic): N = 34
For Cohen's d = 0.5 (moderate): N = 64
For Cohen's d = 0.3 (conservative): N = 176
```

**Decision:** Use N = 50 per condition for Phase 1, re-evaluate for Phase 2 based on observed effect size.

---

## Test Matrix

### Phase 1: Calibration

| System | Nodes | Replications | Total Samples |
|--------|-------|--------------|---------------|
| 1D chain | 100 | 50 | 50 |
| 2D lattice (10×10) | 100 | 50 | 50 |
| 3D lattice (5×5×4) | 100 | 50 | 50 |

**Total Phase 1:** 150 samples

### Phase 2: Compression Tests (if Phase 1 passes)

| System | Condition | Nodes | Replications |
|--------|-----------|-------|--------------|
| 2D lattice | Baseline (no modification) | 100 | 50 |
| 2D lattice | Global boost (ε=0.3) | 100 | 50 |
| 2D lattice | Random noise (σ=0.1) | 100 | 50 |
| 2D lattice | Distance decay (α=0.1) | 100 | 50 |

**Total Phase 2:** 200 samples

### Phase 3: Effect Size (if Phase 2 passes)

Bootstrap analysis on Phase 2 data - no additional samples.

---

## Mandatory Diagnostic Outputs

All outputs are **BLOCKING** - experiment marked incomplete if missing.

### Phase 1 Outputs

| Output | Format | Purpose |
|--------|--------|---------|
| `calibration_results.json` | JSON | All δ values, method agreement |
| `lattice_2d_embedding.png` | PNG | Visual check of 2D embedding |
| `lattice_3d_embedding.png` | PNG | Visual check of 3D embedding |
| `method_comparison.png` | PNG | Isomap vs MDS scatter plot |
| `calibration_gate_status.json` | JSON | Pass/fail for each gate |

### Phase 2 Outputs

| Output | Format | Purpose |
|--------|--------|---------|
| `compression_results.json` | JSON | All δ values by condition |
| `compression_by_condition.png` | PNG | Box plot of δ by condition |
| `effect_ordering.png` | PNG | Verify boost > decay > noise |
| `error_curves.png` | PNG | Reconstruction error vs dimension |

### Phase 3 Outputs

| Output | Format | Purpose |
|--------|--------|---------|
| `effect_size_analysis.json` | JSON | Cohen's d, power, required N |
| `power_curve.png` | PNG | Power vs N at observed effect |
| `confidence_intervals.png` | PNG | 95% CI for each condition |

---

## Distance Transformation

### Calibration Approach

The key issue from Exp01 was δ ≠ 0 for matched systems. We will test multiple distance transformations and select the one that achieves calibration:

**Candidate Transformations:**

1. **Standard:** D = √(2(1 - C))
2. **Linear:** D = 1 - C
3. **Arccos:** D = arccos(C)
4. **Negative log:** D = -log(C + ε)

**Selection Criterion:** Use the transformation that achieves C1-C3 (δ < 0.05 on all lattices).

If NO transformation achieves calibration, report this as a fundamental limitation.

---

## Method Agreement Protocol

**Gate:** If |d_isomap - d_mds| > 0.5, the sample is EXCLUDED.

**Rationale:** Disagreeing methods indicate unreliable dimension estimation for that sample.

**Reporting:** Report exclusion rate. If > 20% of samples excluded, investigate methodology.

---

## Execution Protocol

```
1. PHASE 1: CALIBRATION
   a. Generate 1D, 2D, 3D lattices (N=50 each)
   b. Test all distance transformations
   c. Select best transformation
   d. Evaluate calibration gates C1-C5
   e. Generate mandatory outputs

   IF ANY GATE FAILS:
      → Document failure mode
      → Attempt fix (e.g., different transformation)
      → If unfixable: STOP, report "calibration_failed"

   IF ALL GATES PASS:
      → Proceed to Phase 2

2. PHASE 2: COMPRESSION TESTS
   a. Generate 2D lattices with 4 conditions (N=50 each)
   b. Compute δ for each sample
   c. Evaluate predictions P2.1-P2.4
   d. Generate mandatory outputs

   Report results regardless of pass/fail.

3. PHASE 3: EFFECT SIZE
   a. Compute Cohen's d for compression conditions
   b. Bootstrap 95% confidence intervals
   c. Calculate required N for power=0.8
   d. Generate mandatory outputs
```

---

## Success Criteria

### Minimum Success (Proceed to Exp06)
- All Phase 1 calibration gates pass
- At least 2/4 Phase 2 predictions pass
- Effect size Cohen's d > 0.3

### Full Success
- All Phase 1 gates pass
- All Phase 2 predictions pass
- Cohen's d > 0.5
- Required N < 50

### Failure Modes

| Outcome | Interpretation | Action |
|---------|----------------|--------|
| Phase 1 fails | Method fundamentally broken | Major revision needed |
| Phase 1 passes, Phase 2 fails | No compression effect | Consider archiving hypothesis |
| Phases 1-2 pass, d < 0.3 | Effect too small to measure reliably | Need different approach |

---

## Module Structure

```
experiments/05-calibrated-foundations/
├── spec.md                      # This document
├── claims.json                  # Pre-registered predictions (LOCKED)
├── verification_gates.json      # Calibration and validation gates
├── src/
│   ├── __init__.py
│   ├── lattice_generation.py    # Generate 1D/2D/3D lattices
│   ├── distance_transforms.py   # Multiple D(C) transformations
│   ├── dimension_extraction.py  # Isomap + MDS with agreement check
│   ├── calibration.py           # Phase 1 calibration logic
│   ├── compression_tests.py     # Phase 2 test logic
│   ├── effect_size.py           # Phase 3 analysis
│   ├── diagnostics.py           # Generate all mandatory plots
│   ├── main.py                  # Orchestrate all phases
│   └── tests/
│       ├── __init__.py
│       ├── test_lattices.py
│       ├── test_transforms.py
│       └── test_dimension.py
├── output/
│   ├── phase1/                  # Calibration outputs
│   ├── phase2/                  # Compression test outputs
│   └── phase3/                  # Effect size outputs
├── reports/
│   └── (all mandatory plots)
└── paper/
    └── main.tex
```

---

## Reuse from Previous Experiments

| Module | Source | Modifications |
|--------|--------|---------------|
| Dimension extraction | Exp01/dimension.py | Add method agreement gate |
| Visualization | Exp01/visualization.py | Add new plot types |
| Correlation generation | Exp01/correlation.py | Simplify for lattices |

**New implementations:**
- `lattice_generation.py` - Regular lattice generation
- `distance_transforms.py` - Multiple transformation options
- `calibration.py` - Calibration gate logic
- `effect_size.py` - Cohen's d, power analysis

---

## Timeline

| Phase | Estimated Duration |
|-------|-------------------|
| Implementation | 2-3 hours |
| Phase 1 Execution | 30 minutes |
| Phase 1 Review | 30 minutes |
| Phase 2 Execution (if Phase 1 passes) | 30 minutes |
| Phase 3 Analysis | 30 minutes |
| Documentation & Paper | 1-2 hours |

---

## Commitment

By executing this experiment, we commit to:

1. **NOT adjusting thresholds** - Results stand as measured
2. **Stopping at Phase 1 failure** - No proceeding without calibration
3. **Reporting all failures honestly** - No selective reporting
4. **Generating all mandatory outputs** - Incomplete = failed
5. **Accepting the verdict** - If compression effect is not found, acknowledge it

---

*This specification incorporates all protocol improvements from the Phase 8 Meta-Analysis.*
