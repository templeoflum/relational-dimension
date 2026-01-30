"""
Experiment 01: Graph Topology vs Correlation Structure Dimension

Modules:
- graph_generation: Create RGG and lattice graphs
- correlation: Generate NN, LR, RAND correlation matrices
- dimension: Extract d_topo and d_corr via manifold learning
- metrics: Compute compression ratios and statistics
- visualization: Diagnostic plots and heatmaps
"""

__version__ = "0.1.0"

from . import graph_generation
from . import correlation
from . import dimension
from . import metrics
from . import visualization
