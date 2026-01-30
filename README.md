# Relational Dimension

**Research Question:** Does quantum correlation structure provide geometric information that causal/topological structure alone cannot capture?

**Hypothesis:** Dimension emerges from relational structure between interaction points. Different types of relations (causal order vs quantum correlation) may reveal different dimensional properties.

**Status:** Active research (January 2026)

---

## Motivation

Multiple quantum gravity approaches (Causal Set Theory, Causal Dynamical Triangulations, Asymptotic Safety) converge on dimensional reduction from d≈4 at macroscopic scales to d≈2 at Planck scale. However:

- **Causal Set Theory** extracts dimension purely from causal ordering (partial order ≺)
- **Holographic principle** suggests information capacity scales with area (2D), not volume (3D)
- **Quantum entanglement** creates correlations between spacelike-separated events invisible to causal analysis

This project explores whether **quantum correlation structure** (mutual information, entanglement entropy) reveals geometric properties that pure causal/topological analysis misses.

---

## Theoretical Framework

**Core Insight (from theoretical substrate):**  
Physical reality may be a 2D substrate undergoing recursive correlation processes that generate apparent higher-dimensional structure - similar to how diffusion models denoise images or video feedback loops create depth from stacked 2D planes.

**Key Concepts:**
- **Relational System:** R = (E, ρ, A) where E = events, ρ = relation function, A = observables
- **Causal Dimension d_C:** Extracted from topological/causal relations
- **Quantum Dimension d_Q:** Extracted from correlation/entanglement structure
- **Central Prediction:** For systems with long-range correlations, d_Q < d_C

See `docs/theoretical_substrate.md` for full formalization.

---

## Experiment Series

| ID | Title | Status | Key Finding |
|----|-------|--------|-------------|
| 01 | [Topology vs Correlation Dimension](experiments/01-topology-vs-correlation/) | Specified | Testing classical graphs first |
| 02 | Quantum Lattice Entanglement | Planned | Extend to quantum systems |
| 03 | Scale-Dependent Dimensionality | Planned | Test dimensional running |

---

## Methodology

This project follows [Scientific Collaboration Protocol v1.0](docs/scientific_collaboration_protocol_v1.0.md):

- **Falsification-first:** Every experiment has numbered predictions with quantitative pass/fail thresholds
- **Isolation-before-integration:** Each experiment is self-contained
- **Machine verification:** All quantitative claims verified against source-of-truth data
- **Provenance tracking:** Full Git history + structured DEVLOGs

---

## Quick Start

### Setup
```bash
git clone [repo-url]
cd relational-dimension
pip install -r requirements.txt
```

### Run Experiment 01
```bash
cd experiments/01-topology-vs-correlation
python src/main.py
```

### Verify Claims
```bash
python scripts/verify_claims.py experiments/01-topology-vs-correlation
```

### Check Gates
```bash
python scripts/verify_gates.py check experiments/01-topology-vs-correlation
```

---

## Project Structure

```
relational-dimension/
├── docs/                    # Theoretical framework, protocols
├── scripts/                 # Verification and utility scripts
├── experiments/             # Self-contained experiment directories
│   └── 01-*/
│       ├── spec.md         # Hypothesis, predictions, test matrix
│       ├── DEVLOG.md       # Development log (session-by-session)
│       ├── README.md       # Results summary
│       ├── claims.json     # Machine-verifiable claims manifest
│       ├── src/            # Implementation
│       ├── output/         # Metrics JSON, diagnostic images
│       ├── reports/        # Generated figures, analysis
│       └── paper/          # LaTeX paper
└── FINDINGS.md             # Cross-experiment knowledge tracker
```

---

## Key Results

*To be populated as experiments complete.*

---

## Contributing

This is currently a solo research project (Hedawn + Claude). If you're interested in collaboration or have questions about the methodology, see the protocol document.

---

## License

Research artifacts: CC-BY-4.0  
Code: MIT  

---

**Last Updated:** 2026-01-29  
**Contact:** [Your contact info]
