# Experiment 01: Graph Topology vs Correlation Structure Dimension

**Experiment ID:** 01  
**Status:** Specification  
**Created:** 2026-01-29  
**Protocol Version:** 1.0

---

## 1. Research Question

Does correlation structure between nodes reveal geometric information that graph topology alone cannot capture?

---

## 2. Hypothesis

**Primary Claim:** When nodes have correlation patterns that deviate from their topological connectivity, the correlation-based dimensional embedding will differ from the topology-based dimensional embedding.

**Mechanistic Prediction:** Strong correlations between topologically distant nodes create "shortcuts" in correlation space, allowing lower-dimensional embedding than topology suggests.

---

## 3. Numbered Predictions with Thresholds

### P1: Baseline Agreement
**Claim:** For graphs where correlation structure matches topology (nearest-neighbor correlations only), topological and correlation dimensions agree.

**Measurement:** d_topo vs d_corr for NN-correlated random geometric graphs

**Threshold:** |d_topo - d_corr| < 0.2

**Rationale:** When correlation follows topology exactly, both methods should extract the same dimensional structure. This validates our measurement approach.

---

###  P2: Long-Range Compression
**Claim:** For graphs with long-range correlations, correlation dimension is significantly lower than topological dimension.

**Measurement:** Compression ratio δ = (d_topo - d_corr) / d_topo for LR-correlated graphs

**Threshold:** δ > 0.25

**Rationale:** This is the core prediction - that correlation shortcuts allow geometric compression invisible to pure topology.

**Falsification:** If δ ≤ 0.15 for all LR configurations, the hypothesis fails.

---

### P3: Scaling Behavior
**Claim:** Compression effect increases with graph size for fixed correlation density.

**Measurement:** δ(N=50) vs δ(N=200) for identical correlation patterns scaled proportionally

**Threshold:** δ₂₀₀ / δ₅₀ > 1.3

**Rationale:** If this is a genuine geometric effect, larger systems provide more opportunity for compression. If it's a finite-size artifact, the ratio approaches 1.

---

### P4: Random Control
**Claim:** Random correlations (no structure) do not produce systematic compression.

**Measurement:** δ for random correlation matrices vs structured LR correlations

**Threshold:** δ_random < 0.15 AND δ_LR > 0.25

**Rationale:** Structure matters, not just correlation density. Random correlations shouldn't create geometric shortcuts.

---

### P5: Correlation Strength Dose-Response
**Claim:** Compression increases monotonically with long-range correlation strength.

**Measurement:** δ vs correlation strength α ∈ [0, 0.2, 0.4, 0.6, 0.8, 1.0]

**Threshold:** Pearson correlation r(α, δ) > 0.8

**Rationale:** Should see smooth dose-response curve if effect is real. Noisy/non-monotonic suggests measurement artifacts.

---

## 4. Test Matrix

### 4.1 Graph Types

| Graph Type | N (nodes) | Topology | Correlation Pattern | Purpose |
|-----------|-----------|----------|---------------------|---------|
| RGG-NN | 50, 100, 200 | Random geometric, radius r | Only NN pairs correlated | Baseline (P1) |
| RGG-LR | 50, 100, 200 | Random geometric, radius r | NN + diagonal/skip-2 correlations | Core test (P2, P3) |
| RGG-RAND | 50, 100, 200 | Random geometric, radius r | Random pair correlations | Control (P4) |
| Lattice-NN | 10×10, 15×15, 20×20 | 2D square lattice | Only NN correlations | Baseline validation |
| Lattice-LR | 10×10, 15×15, 20×20 | 2D square lattice | NN + long diagonals | Structured test |

### 4.2 Correlation Patterns

**Nearest-Neighbor (NN):**
- C[i,j] = ρ if d_graph(i,j) = 1, else 0
- ρ = correlation strength ∈ [0.5, 1.0]

**Long-Range (LR):**
- C[i,j] = ρ · exp(-d_euclidean(i,j) / λ)
- λ = correlation length, set to 0.3 × graph_diameter
- Creates correlations across topologically distant nodes

**Random (RAND):**
- Select k random pairs, set C[i,j] = ρ
- k chosen to match total correlation density of LR

### 4.3 Parameters

- **Graph sizes:** N ∈ {50, 100, 200} for random graphs; side ∈ {10, 15, 20} for lattices
- **Correlation strength:** α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}
- **Replications:** 10 random graphs per configuration
- **Embedding methods:** Isomap, MDS (both methods must agree within 0.3 for result to be valid)

---

## 5. Measurement Procedures

### 5.1 Topological Dimension (d_topo)

**Method:** Graph geodesic distance embedding

1. Compute all-pairs shortest path distances: D_topo[i,j] = graph_distance(i,j)
2. Apply Isomap with n_neighbors = 8
3. Compute reconstruction error vs embedding dimension k: err(k)
4. Define d_topo = minimum k where err(k) < 0.1 · err(k=1)

**Validation:** Compare Isomap vs classical MDS. If they differ by >0.5, flag as unstable.

### 5.2 Correlation Dimension (d_corr)

**Method:** Correlation-based distance embedding

1. Convert correlation matrix to distance: D_corr[i,j] = sqrt(2(1 - C[i,j]))
2. Apply Isomap with n_neighbors = 8
3. Compute reconstruction error vs embedding dimension k
4. Define d_corr = minimum k where err(k) < 0.1 · err(k=1)

**Validation:** Same as topological - require Isomap and MDS agreement.

### 5.3 Compression Ratio

**δ = (d_topo - d_corr) / d_topo**

- δ > 0: Correlation dimension is lower (compression)
- δ ≈ 0: Dimensions agree
- δ < 0: Correlation dimension is higher (expansion - unexpected)

---

## 6. Baselines and Controls

### 6.1 Identity Baseline
**Graph:** Complete graph (all nodes connected)  
**Correlation:** Uniform C[i,j] = 0.5 for all i≠j  
**Expected:** d_topo = d_corr = 0 (all nodes equidistant)  
**Purpose:** Verify measurement isn't producing spurious dimensions

### 6.2 Known Geometry
**Graph:** 2D lattice with only NN connections  
**Correlation:** C[i,j] = 1 if NN, else 0  
**Expected:** d_topo = d_corr ≈ 2.0  
**Purpose:** Verify we recover known dimensionality

### 6.3 Null Model
**Graph:** Random geometric graph  
**Correlation:** No correlations (C = I, identity matrix)  
**Expected:** d_corr = N-1 (maximal dimension)  
**Purpose:** Verify correlation structure matters

---

## 7. Implementation Notes

### 7.1 Graph Generation

```python
# Random Geometric Graph
positions = np.random.rand(N, 2)
distances = scipy.spatial.distance.cdist(positions, positions)
adjacency = (distances < radius).astype(float)

# 2D Lattice
G = nx.grid_2d_graph(side, side)
```

### 7.2 Correlation Generation

```python
# Long-range correlations
euclidean_dist = scipy.spatial.distance.cdist(positions, positions)
C = rho * np.exp(-euclidean_dist / lambda_corr)
np.fill_diagonal(C, 1.0)  # Self-correlation = 1
```

### 7.3 Dimension Extraction

```python
from sklearn.manifold import Isomap, MDS

# Topological
isomap_topo = Isomap(n_components=k_max, metric='precomputed')
embedding = isomap_topo.fit_transform(D_topo)
errors_topo = [reconstruction_error(D_topo, embedding[:, :k]) 
               for k in range(1, k_max+1)]

# Find knee in error curve
d_topo = np.argmax(errors_topo < threshold) + 1
```

---

## 8. Diagnostic Outputs

For each test configuration, generate:

1. **Correlation heatmaps:** C[i,j] visualization showing pattern structure
2. **Distance comparison plots:** D_topo vs D_corr scatter plots
3. **Embedding visualizations:** 2D projections of both dimensional spaces
4. **Error curves:** Reconstruction error vs embedding dimension for both methods
5. **Compression scatter:** δ vs graph size, δ vs correlation strength

---

## 9. Success Criteria

**Full success:** All 5 predictions pass their thresholds.

**Partial success scenarios:**
- P1 passes, P2-P5 fail: Measurement works, but hypothesis is wrong
- P1 fails: Measurement approach is broken, redesign needed
- P2 passes, P3 fails: Effect exists but doesn't scale (finite-size artifact)
- P4 fails: Structure doesn't matter, only density (contradicts hypothesis)

**Falsification:** If P2 fails (no compression for long-range correlations), the core hypothesis is falsified.

---

## 10. Computational Constraints

- **Memory:** O(N²) for distance matrices, manageable up to N=500 on 32GB
- **Time:** Isomap is O(N³) worst case, but with n_neighbors=8 should be ~O(N² log N)
- **Estimate:** Each graph configuration ~5 seconds, full test matrix ~30 minutes
- **Feasibility:** 100% within laptop constraints

---

## 11. Expected Timeline

- **Implementation:** 4-6 hours (graph generation, correlation functions, dimension extraction, plotting)
- **Execution:** 30 minutes (automated sweep over test matrix)
- **Analysis:** 2-3 hours (generate figures, interpret results, draft findings)
- **Documentation:** 2-3 hours (DEVLOG, paper sections, README)

**Total:** Single session (8-12 hours) or two half-sessions.

---

## Document Metadata

**Experiment:** 01  
**Title:** Graph Topology vs Correlation Structure Dimension  
**Status:** Specification Complete  
**Next Phase:** Implementation (Phase 2)  
**Dependencies:** None (first experiment in series)
