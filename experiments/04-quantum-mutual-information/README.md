# Experiment 04: Quantum Mutual Information and Dimensional Compression

## Overview

This experiment pivots from classical correlations to **actual quantum systems**. We test whether quantum mutual information reveals dimensional compression invisible to topological analysis.

This is the first experiment to use genuine quantum states rather than classical proxies.

## Research Question

Does quantum entanglement create dimensional compression measurable via mutual information geometry?

## Quick Start

```bash
# Run experiment
cd experiments/04-quantum-mutual-information
python src/main.py

# Run tests
pytest src/tests/ -v
```

## Predictions

| ID | Description | Threshold | Status | Measured |
|----|-------------|-----------|--------|----------|
| P1 | Product State Baseline | \|delta\| < 0.1 | FAIL | 0.889 |
| P2 | GHZ Compression | delta > 0.4 | PASS | 0.824 |
| P3 | Cluster State Topology | d_Q < 1.5 | FAIL | 3.536 |
| P4 | Random State Compression | delta > 0.25 | PASS | 0.472 |
| P5 | Entanglement-Compression Correlation | r > 0.7 | FAIL | -0.526 |
| P6 | 2D Holographic Prediction | d_Q < 1.5 | PASS | 1.000 |

**Result: 3/6 predictions passed**

## Key Finding

Uniform correlation patterns (whether from zero entanglement or maximal global entanglement) collapse to 1D embeddings. MI heterogeneity, not entanglement magnitude, determines effective dimensionality.

## Quantum States Tested

1. **Product State** - No entanglement (baseline)
2. **GHZ State** - Maximal global entanglement
3. **W State** - Symmetric entanglement
4. **1D Cluster State** - Topological entanglement on chain
5. **2D Cluster State** - Topological entanglement on grid
6. **Haar Random States** - Typical highly entangled states

## Test Matrix

| N (qubits) | Geometry | Memory | Purpose |
|------------|----------|--------|---------|
| 4 | Chain | 4 KB | Fast baseline |
| 6 | Chain | 64 KB | Small system |
| 8 | Chain | 1 MB | Medium 1D |
| 8 | 2x4 grid | 1 MB | Small 2D |
| 10 | Chain | 16 MB | Larger 1D |
| 12 | Chain | 256 MB | Tractability limit |
| 12 | 3x4 grid | 256 MB | 2D at limit |

## Key Formulas

**Quantum Mutual Information:**
```
I(A:B) = S(rho_A) + S(rho_B) - S(rho_AB)
```

**Von Neumann Entropy:**
```
S(rho) = -Tr(rho log rho)
```

**Compression Ratio:**
```
delta = (d_topo - d_Q) / d_topo
```

## Output Files

- `output/metrics.json` - Full experiment results
- `reports/mi_heatmaps.png` - MI matrices for each state type
- `reports/compression_by_state.png` - delta by state type
- `reports/entanglement_correlation.png` - S_ent vs delta
- `reports/predictions_summary.png` - P1-P6 results
- `paper/main.pdf` - Compiled paper

## Dependencies

- numpy, scipy, matplotlib
- Optional: qutip (for verification)

## Computational Constraints

Full density matrix simulation:
- N qubits → 2^N dimensional Hilbert space
- Memory: 2^(2N) × 16 bytes
- Feasible limit: N ≤ 12 qubits on 16 GB RAM
