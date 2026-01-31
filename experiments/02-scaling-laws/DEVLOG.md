# Development Log: Experiment 02

## Session 1: Initial Implementation

### Completed
- Created experiment directory structure
- Implemented `dimension_improved.py`:
  - `continuous_dimension()` - fractional dimension via error curve interpolation
  - `landmark_mds()` - sparse MDS for large N
  - `sparse_isomap()` - k-NN based Isomap
  - `compute_reconstruction_errors()` - same approach as Exp01 for consistency
  - `find_fractional_dimension_from_errors()` - linear interpolation on error curves
- Implemented `scaling_analysis.py`:
  - `fit_log_model()` - d = a*log(N) + b
  - `fit_power_model()` - d = c*N^alpha
  - `predict_delta()` - extrapolation
  - `compare_models()` - select best fit
  - `evaluate_*()` - P1-P6 prediction evaluators
- Implemented `main.py`:
  - Test matrix runner
  - Model fitting pipeline
  - Visualization generation
- Created unit tests for all modules (48 tests, all passing)

### Design Decisions
1. **Continuous dimension**: Using error threshold 0.1 (same as Exp01) with linear interpolation for fractional output
2. **Sparse methods**: Landmark MDS with 200 landmarks for N>500, k-NN Isomap with k=15
3. **Training split**: Fit models on N ≤ 500, test on N > 500 for P5
4. **Visualization**: Log-scale x-axis for scaling curves

### Initial Results
First run completed:
- N=50-500: positive delta (0.11 to 0.39), consistent with Exp01
- N=750-1000: negative delta (-0.15 to -0.10), possible sparse method artifact
- Log model R² = 0.81 (close to 0.85 threshold)
- All 6 predictions failed due to discontinuity at N>500

### Key Findings
1. Scaling IS visible in N=50-500 range, consistent with Exp01
2. Sparse methods for N>500 produce qualitatively different behavior
3. High variance at some sizes suggests need for more replications
4. Log model fits better than power law (R² 0.81 vs 0.63)

### Next Steps (potential)
- Investigate sparse vs full method transition
- Consider using full methods for all sizes (memory permits)
- Increase replications for better confidence intervals
