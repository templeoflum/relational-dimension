# Development Log: Experiment 01

**Experiment:** Topology vs Correlation Structure Dimension  
**Started:** 2026-01-29

---

## Session 1: 2026-01-29

### Context
First experiment in the relational-dimension series. Testing whether correlation structure reveals geometric information that pure topology misses.

### Goals
- [x] Implement graph generation (RGG, lattice)
- [x] Implement correlation pattern generation (NN, LR, RAND)
- [x] Implement dimension extraction (Isomap, MDS)
- [x] Create workbench that runs full test matrix
- [x] Generate diagnostic outputs
- [x] Create unit tests

### Progress

**Completed implementation of all modules:**

1. **graph_generation.py** - Functions for:
   - `create_rgg(n, radius, seed)` - Random geometric graphs
   - `create_lattice(side)` - 2D square lattice
   - `get_positions()`, `get_adjacency()`, `get_graph_distances()`
   - `get_largest_component()` - Handle disconnected graphs
   - `estimate_radius()` - Estimate RGG radius for target degree

2. **correlation.py** - Functions for:
   - `correlation_nn(adjacency, rho)` - Nearest-neighbor only
   - `correlation_lr(positions, rho, lambda_corr)` - Long-range exponential
   - `correlation_rand(n, k, rho)` - Random pairs
   - `correlation_to_distance(C)` - D = sqrt(2(1-C))
   - `ensure_positive_semidefinite(C)` - Eigenvalue clipping

3. **dimension.py** - Functions for:
   - `extract_dimension_isomap(D, k_max, n_neighbors)`
   - `extract_dimension_mds(D, k_max)`
   - `compute_reconstruction_errors()` - Normalized RMSE
   - `find_intrinsic_dimension(errors, threshold)` - Knee detection
   - `extract_dimension_both()` - Combined with validation

4. **metrics.py** - Functions for:
   - `compression_ratio(d_topo, d_corr)` - delta calculation
   - `aggregate_results()` - Mean, std across replications
   - `evaluate_p1()` through `evaluate_p5()` - Prediction checks
   - `summarize_predictions()` - Pass/fail summary

5. **visualization.py** - Functions for:
   - Correlation heatmaps
   - Distance comparison scatter plots
   - 2D embedding visualizations
   - Error curve plots
   - Compression scatter and dose-response curves
   - Prediction summary bar charts

6. **main.py** - Test matrix runner:
   - Runs baselines (known geometry, identity)
   - Runs full test matrix (RGG/lattice x NN/LR/RAND)
   - Runs dose-response experiment
   - Evaluates all 5 predictions
   - Saves metrics.json with full results
   - Generates all diagnostic PNGs

7. **tests/** - Unit and integration tests:
   - test_graph_generation.py
   - test_correlation.py
   - test_dimension.py
   - test_integration.py

### Findings

**Experiment Results (2026-01-29):**

The core hypothesis was **FALSIFIED**. Only 1 of 5 predictions passed.

| Prediction | Result | Measured | Threshold |
|------------|--------|----------|-----------|
| P1 (Baseline) | FAIL | 0.368 | < 0.2 |
| P2 (Compression) | FAIL | 0.246 | > 0.25 |
| P3 (Scaling) | PASS | 3.11 | > 1.3 |
| P4 (Control) | FAIL | rand=0.11, lr=0.25 | rand<0.15 AND lr>0.25 |
| P5 (Dose-Response) | FAIL | r=0.18 | > 0.8 |

**Key Observations:**

1. **P1 Failure (Baseline Agreement)**: Even for nearest-neighbor correlations, dimension estimates show high variance. The topological and correlation dimensions don't agree well (|delta| = 0.37 vs threshold 0.2). This indicates measurement instability.

2. **P2 Near-Miss**: Long-range compression measured at delta = 0.246, just barely below the 0.25 threshold. The effect exists but is weaker than predicted.

3. **P3 Success (Scaling)**: The compression ratio does increase with graph size (ratio = 3.11 > 1.3). Larger graphs show more compression, suggesting the effect is real but perhaps the magnitude thresholds were too aggressive.

4. **High Variance in RGG Results**: The delta values for RGG show enormous variance (e.g., -0.75 to +0.67 for N=50). This is likely due to:
   - Isomap/MDS disagreement on dimension estimates
   - Small sample sizes relative to noise
   - Sensitivity to graph connectivity

5. **Lattice Results More Stable**: Lattice graphs show deterministic results (zero std dev) because the graph structure is fixed. This is a useful validation that the measurement pipeline works.

6. **Baseline Calibration Issue**: The "known geometry" baseline shows d_topo=2.0 but d_corr=3.0, giving delta=-0.5. This indicates the correlation-to-distance transformation may not preserve the expected dimensionality.

**Interpretations:**

- The original hypothesis that long-range correlations create "geometric shortcuts" may be correct in principle, but our measurement approach lacks the precision to detect it reliably.
- The Isomap vs MDS disagreement (often differing by 2-6 dimensions) suggests the dimension estimation method needs refinement.
- Alternative interpretation: correlation structure provides *different* geometric information, not necessarily *compressed* information.

**Recommended Next Steps:**

1. Investigate why Isomap and MDS give different dimension estimates
2. Try alternative dimension estimation methods (PCA explained variance, correlation dimension)
3. Consider using fractional dimension estimates instead of threshold-based detection
4. Increase replications or use bootstrap confidence intervals
5. Test with stronger correlation patterns (higher rho or different decay functions)

### Process Notes

- Used eigenvalue clipping for PSD projection of correlation matrices
- Normalized distances for scale-invariance in dimension estimation
- Isomap and MDS cross-validation catches unstable estimates
- Lattice sides chosen to approximate target node counts: 7x7=49, 10x10=100, 14x14=196

### Next Steps

1. ~~Run `pip install -r requirements.txt` to install dependencies~~ DONE
2. ~~Run `pytest src/tests/` to verify implementation~~ DONE (68/68 passed)
3. ~~Run `python src/main.py` to execute full experiment~~ DONE
4. ~~Review output/metrics.json and diagnostic PNGs~~ DONE
5. Decide whether to refine methodology or accept falsification
6. If refining: focus on dimension estimation stability
7. If accepting: document lessons learned and move to next experiment

---

## Session Template (copy for new sessions)

## Session N: YYYY-MM-DD

### Context
*What happened previously? What's the state of the experiment?*

### Goals
- [ ] Goal 1
- [ ] Goal 2

### Progress
*What was actually done? What worked? What didn't?*

### Findings
*Any new discoveries, failed predictions, or insights?*

### Process Notes
*Observations about the methodology itself?*

### Next Steps
*What needs to happen next?*

---
