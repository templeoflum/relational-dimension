# Graph Topology vs. Correlation Structure: Does Correlation Reveal Hidden Geometry?

**A Falsifiable Experiment Report**

Experiment 01 --- Relational Dimension Project
January 29, 2026

---

## Abstract

We test the hypothesis that correlation structure between graph nodes reveals geometric information that pure topological connectivity cannot capture. Specifically, we predict that graphs with long-range correlations (correlations between topologically distant nodes) will exhibit lower effective dimensionality in correlation space than in topological space. We formalize this as five falsifiable predictions with pre-specified thresholds. After testing 150 configurations across random geometric graphs and lattices with three correlation patterns, we find that **4 of 5 predictions fail**, including the core hypothesis. The measured compression ratio for long-range correlations (δ = 0.246) falls just below the pre-specified threshold of 0.25. We discuss methodological limitations, particularly the instability of dimension estimation, and propose refinements for future work.

---

## 1. Introduction

Graph-based representations are ubiquitous in modeling complex systems, from social networks to neural connectivity. A fundamental question is whether the *correlation structure* between nodes—how their states or activities co-vary—contains geometric information beyond what the graph's *topological structure* (edges and paths) reveals.

### Core Hypothesis

When nodes have correlation patterns that deviate from their topological connectivity, the correlation-based dimensional embedding will differ from the topology-based embedding. Specifically, strong correlations between topologically distant nodes create "shortcuts" in correlation space, potentially allowing lower-dimensional representations.

### Why This Matters

If correlation structure reveals latent geometry invisible to topology, this has implications for:
- Dimensionality reduction in network analysis
- Detecting hidden structure in complex systems
- Understanding the relationship between functional and structural connectivity

---

## 2. Methods

### 2.1 Graph Types

**Random Geometric Graphs (RGG):** Nodes are placed uniformly at random in [0,1]², with edges connecting pairs within radius r. The radius is chosen to achieve average degree ≈ 6.

**Square Lattices:** Regular 2D grids with nearest-neighbor connectivity. Lattices provide a known-geometry baseline where we expect d ≈ 2.

### 2.2 Correlation Patterns

**Nearest-Neighbor (NN):** C_ij = ρ if nodes i,j are adjacent, else C_ij = 0 (plus identity diagonal). This serves as a baseline where correlation matches topology.

**Long-Range (LR):** C_ij = ρ · exp(-d_euclidean(i,j) / λ), where λ = 0.3 × diameter. This creates correlations based on geometric proximity rather than topological distance.

**Random (RAND):** k random node pairs receive correlation ρ, where k matches the number of significant correlations in the LR pattern. This controls for correlation density without geometric structure.

### 2.3 Dimension Extraction

**Topological Dimension (d_topo):**
1. Compute all-pairs shortest path distances D_topo
2. Apply Isomap embedding with n_neighbors = 8
3. Find minimum k where reconstruction error < 0.1 × initial error

**Correlation Dimension (d_corr):**
1. Convert correlation to distance: D_ij = √(2(1 - C_ij))
2. Apply same Isomap procedure

### 2.4 Compression Ratio

The key metric is:

**δ = (d_topo - d_corr) / d_topo**

- δ > 0: Correlation dimension is lower (compression)
- δ ≈ 0: Dimensions agree
- δ < 0: Correlation dimension is higher (expansion)

### 2.5 Test Matrix

- Graph sizes: N ∈ {50, 100, 200} for RGG; sides ∈ {7, 10, 14} for lattices
- Correlation strength: ρ = 0.8
- Replications: 10 per configuration
- Total: 150 configurations

---

## 3. Pre-Registered Predictions

We state five predictions with explicit pass/fail thresholds, determined *before* running the experiment.

| ID | Prediction | Pass Criterion | Fail Criterion |
|----|------------|----------------|----------------|
| P1 | Baseline Agreement | \|δ_NN\| < 0.2 | \|δ_NN\| ≥ 0.2 |
| P2 | LR Compression | δ_LR > 0.25 | δ_LR ≤ 0.15 |
| P3 | Scaling | δ_200 / δ_50 > 1.3 | ratio ≤ 1.0 |
| P4 | Control | δ_rand < 0.15 AND δ_LR > 0.25 | δ_rand ≥ δ_LR |
| P5 | Dose-Response | r(α, δ) > 0.8 | r < 0.5 |

**P1 (Baseline Agreement):** When correlation follows topology exactly (NN pattern), topological and correlation dimensions should agree. This validates our measurement approach.

**P2 (Core Hypothesis):** Long-range correlations should produce significant compression (δ > 0.25). This is the central claim—that correlation shortcuts enable lower-dimensional embedding.

**P3 (Scaling):** The compression effect should increase with graph size, ruling out finite-size artifacts.

**P4 (Control):** Random correlations should not produce systematic compression, demonstrating that geometric structure matters.

**P5 (Dose-Response):** Compression should increase monotonically with correlation strength α, showing a smooth causal relationship.

---

## 4. Results

### 4.1 Summary

| ID | Description | Threshold | Measured | Result |
|----|-------------|-----------|----------|--------|
| P1 | Baseline Agreement | < 0.20 | 0.368 | **FAIL** |
| P2 | LR Compression | > 0.25 | 0.246 | **FAIL** |
| P3 | Scaling Behavior | > 1.30 | 3.112 | **PASS** |
| P4 | Random Control | rand < 0.15, LR > 0.25 | 0.11, 0.25 | **FAIL** |
| P5 | Dose-Response | r > 0.80 | 0.175 | **FAIL** |

**Predictions passed: 1/5**

### 4.2 Detailed Results

**P1 Failure: High Baseline Variance**
The mean absolute compression ratio for NN correlations was |δ| = 0.368, far exceeding the 0.2 threshold. Individual measurements ranged from -0.75 to +0.67. This high variance indicates that even when correlation matches topology, our dimension estimates are unstable.

**P2 Near-Miss: Compression Below Threshold**
Long-range correlations produced mean δ = 0.246, just below the 0.25 threshold. While this suggests some compression effect may exist, it fails to meet our pre-specified criterion.

**P3 Success: Scaling Confirmed**
The ratio δ_200/δ_50 = 3.11 strongly exceeds the 1.3 threshold. Compression increases with graph size:
- N = 50: δ = 0.108 ± 0.46
- N = 200: δ = 0.336 ± 0.49

**P4 Failure: LR Threshold Not Met**
While random correlations showed low compression (δ_rand = 0.11 < 0.15), the LR compression (δ_LR = 0.246) failed to exceed 0.25, causing overall failure.

**P5 Failure: No Dose-Response**
The correlation between strength α and compression δ was r = 0.175 (p = 0.74), indicating no significant relationship.

### 4.3 Baseline Calibration Issue

The known-geometry baseline (10×10 lattice with NN correlations) revealed a calibration issue:
- d_topo = 2.0 (expected)
- d_corr = 3.0 (expected 2.0)

This δ = -0.5 indicates the correlation-to-distance transformation does not preserve dimensionality as expected.

### 4.4 Method Disagreement

A key finding is the disagreement between Isomap and MDS:
- For RGG N=200: Isomap gives d = 2.0, MDS gives d = 8.1
- This 6-dimension gap suggests fundamental instability in dimension estimation

---

## 5. Discussion

### 5.1 Interpretation

The core hypothesis—that long-range correlations create geometric shortcuts enabling dimensional compression—is **not supported** by these data. While P2 came close (δ = 0.246 vs. threshold 0.25), we must respect the pre-registered threshold.

However, the P1 failure suggests a deeper problem: our measurement approach may lack the precision to detect the hypothesized effect.

### 5.2 Why Did P3 Pass?

The scaling prediction passed despite other failures. This could indicate:
1. A real but weak effect that becomes detectable only at larger scales
2. Systematic bias in dimension estimation that scales with N
3. Improved Isomap/MDS stability at larger sample sizes

### 5.3 Methodological Limitations

1. **Dimension estimation instability:** Isomap and MDS often disagreed by 2-6 dimensions
2. **Threshold-based detection:** Using a fixed error threshold (0.1) may be too sensitive to noise
3. **Correlation-to-distance transformation:** The D = √(2(1-C)) mapping may not preserve geometric structure
4. **Small graph sizes:** N ≤ 200 may be insufficient for stable manifold learning

---

## 6. Conclusion

We tested whether correlation structure reveals hidden low-dimensional geometry in graphs. Our pre-registered predictions were largely falsified:

- **Core hypothesis (P2):** Not supported (δ = 0.246 < 0.25)
- **Baseline calibration (P1):** Failed, indicating measurement issues
- **Scaling (P3):** Confirmed, suggesting possible weak effect
- **Controls (P4, P5):** Failed

The experiment demonstrates the value of pre-registered predictions: without explicit thresholds, one might interpret δ = 0.246 as "substantial compression." The falsifiable framework forces intellectual honesty.

### Recommendations for Future Work

1. Use continuous dimension estimators (e.g., explained variance ratios) rather than threshold-based detection
2. Require stronger Isomap/MDS agreement before accepting estimates
3. Test alternative manifold learning methods (UMAP, t-SNE with fixed perplexity)
4. Increase graph sizes (N ≥ 500) for more stable estimation
5. Consider fractional dimension estimators

---

## Data Availability

All code, data, and analysis artifacts are available in the repository:
- Source code: `experiments/01-topology-vs-correlation/src/`
- Raw results: `experiments/01-topology-vs-correlation/output/metrics.json`
- Figures: `experiments/01-topology-vs-correlation/output/*.png`
- LaTeX source: `experiments/01-topology-vs-correlation/paper/main.tex`
- Compiled PDF: `experiments/01-topology-vs-correlation/paper/main.pdf`
