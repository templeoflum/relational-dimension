"""
Visualization module for Experiment 01.

Provides functions for diagnostic plots and publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any


def setup_style():
    """Set up matplotlib style for consistent figures."""
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_correlation_heatmap(C: np.ndarray, title: str, filepath: Path,
                             show_colorbar: bool = True):
    """
    Plot correlation matrix as heatmap.

    Args:
        C: Correlation matrix
        title: Plot title
        filepath: Output file path
        show_colorbar: Whether to show colorbar
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_title(title)
    ax.set_xlabel('Node index')
    ax.set_ylabel('Node index')

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_distance_comparison(D_topo: np.ndarray, D_corr: np.ndarray,
                             filepath: Path, title: str = 'Distance Comparison'):
    """
    Scatter plot of topological vs correlation distances.

    Args:
        D_topo: Topological distance matrix
        D_corr: Correlation-based distance matrix
        filepath: Output file path
        title: Plot title
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get upper triangle (exclude diagonal)
    n = D_topo.shape[0]
    triu_idx = np.triu_indices(n, k=1)

    d_topo_flat = D_topo[triu_idx]
    d_corr_flat = D_corr[triu_idx]

    # Normalize for comparison
    d_topo_norm = d_topo_flat / np.max(d_topo_flat) if np.max(d_topo_flat) > 0 else d_topo_flat
    d_corr_norm = d_corr_flat / np.max(d_corr_flat) if np.max(d_corr_flat) > 0 else d_corr_flat

    ax.scatter(d_topo_norm, d_corr_norm, alpha=0.3, s=10)

    # Add diagonal reference line
    lims = [0, 1]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x')

    ax.set_xlabel('Topological Distance (normalized)')
    ax.set_ylabel('Correlation Distance (normalized)')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_embedding_2d(embedding: np.ndarray, filepath: Path,
                      title: str = 'Embedding', labels: Optional[np.ndarray] = None):
    """
    Plot 2D projection of embedding.

    Args:
        embedding: N x k embedding (uses first 2 components)
        filepath: Output file path
        title: Plot title
        labels: Optional node labels for coloring
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    if embedding.shape[1] < 2:
        # If only 1D, plot on line
        ax.scatter(embedding[:, 0], np.zeros(len(embedding)), alpha=0.6, s=30)
    else:
        if labels is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                               c=labels, cmap='viridis', alpha=0.6, s=30)
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=30)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_error_curve(errors: np.ndarray, d_estimate: int, filepath: Path,
                     title: str = 'Reconstruction Error', method: str = ''):
    """
    Plot reconstruction error vs embedding dimension.

    Args:
        errors: Error values for each dimension
        d_estimate: Estimated intrinsic dimension
        filepath: Output file path
        title: Plot title
        method: Method name (Isomap/MDS) for labeling
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    dims = np.arange(1, len(errors) + 1)
    ax.plot(dims, errors, 'b-o', markersize=6, label=f'{method} error')

    # Mark estimated dimension
    if 1 <= d_estimate <= len(errors):
        ax.axvline(x=d_estimate, color='r', linestyle='--', alpha=0.7,
                   label=f'Estimated d={d_estimate}')
        ax.plot(d_estimate, errors[d_estimate - 1], 'ro', markersize=10)

    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Reconstruction Error (normalized)')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(dims)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_error_curves_comparison(errors_isomap: np.ndarray, errors_mds: np.ndarray,
                                 d_isomap: int, d_mds: int, filepath: Path,
                                 title: str = 'Error Curves Comparison'):
    """
    Plot both Isomap and MDS error curves on same axes.

    Args:
        errors_isomap: Isomap error values
        errors_mds: MDS error values
        d_isomap: Isomap dimension estimate
        d_mds: MDS dimension estimate
        filepath: Output file path
        title: Plot title
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    dims = np.arange(1, len(errors_isomap) + 1)

    ax.plot(dims, errors_isomap, 'b-o', markersize=6, label=f'Isomap (d={d_isomap})')
    ax.plot(dims, errors_mds, 'g-s', markersize=6, label=f'MDS (d={d_mds})')

    ax.axvline(x=d_isomap, color='b', linestyle='--', alpha=0.5)
    ax.axvline(x=d_mds, color='g', linestyle='--', alpha=0.5)

    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Reconstruction Error (normalized)')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(dims)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_compression_scatter(results: Dict[str, Any], filepath: Path,
                             title: str = 'Compression Ratio vs Graph Size'):
    """
    Scatter plot of compression ratio vs graph size.

    Args:
        results: Results dictionary with delta values by N
        filepath: Output file path
        title: Plot title
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'rgg_nn': 'blue', 'rgg_lr': 'red', 'rgg_rand': 'gray',
              'lattice_nn': 'cyan', 'lattice_lr': 'orange'}
    markers = {'rgg_nn': 'o', 'rgg_lr': 's', 'rgg_rand': '^',
               'lattice_nn': 'D', 'lattice_lr': 'v'}

    for config_name, config_data in results.items():
        if 'delta_by_n' not in config_data:
            continue

        Ns = []
        deltas_mean = []
        deltas_std = []

        for n, delta_list in sorted(config_data['delta_by_n'].items()):
            Ns.append(n)
            deltas_mean.append(np.mean(delta_list))
            deltas_std.append(np.std(delta_list))

        color = colors.get(config_name, 'black')
        marker = markers.get(config_name, 'o')

        ax.errorbar(Ns, deltas_mean, yerr=deltas_std, fmt=f'{marker}-',
                   color=color, label=config_name, capsize=4, markersize=8)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='P2 threshold')

    ax.set_xlabel('Graph Size (N)')
    ax.set_ylabel('Compression Ratio (delta)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_dose_response(alpha_values: List[float], delta_values: List[float],
                       filepath: Path, r_value: float = None,
                       title: str = 'Dose-Response: Compression vs Correlation Strength'):
    """
    Plot compression ratio vs correlation strength (dose-response curve).

    Args:
        alpha_values: Correlation strength values
        delta_values: Compression ratios
        filepath: Output file path
        r_value: Pearson correlation (optional, for annotation)
        title: Plot title
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alpha_values, delta_values, 'bo-', markersize=8, linewidth=2)

    # Add trend line
    if len(alpha_values) >= 2:
        z = np.polyfit(alpha_values, delta_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(alpha_values), max(alpha_values), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Linear fit')

    if r_value is not None:
        ax.annotate(f'r = {r_value:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, verticalalignment='top')

    ax.set_xlabel('Correlation Strength (alpha)')
    ax.set_ylabel('Compression Ratio (delta)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_predictions_summary(predictions: Dict[str, Dict], filepath: Path,
                            title: str = 'Prediction Results'):
    """
    Bar chart showing pass/fail for each prediction.

    Args:
        predictions: Dictionary of prediction evaluations
        filepath: Output file path
        title: Plot title
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(predictions.keys())
    passed = [1 if predictions[p].get('passed', False) else 0 for p in names]
    colors = ['green' if p else 'red' for p in passed]

    bars = ax.bar(names, [1] * len(names), color=colors, alpha=0.7, edgecolor='black')

    # Add threshold and measured annotations
    for i, (name, pred) in enumerate(predictions.items()):
        measured = pred.get('measured', 'N/A')
        threshold = pred.get('threshold', 'N/A')
        if isinstance(measured, float):
            measured = f'{measured:.3f}'
        if isinstance(threshold, float):
            threshold = f'{threshold:.3f}'
        ax.annotate(f'Measured: {measured}\nThreshold: {threshold}',
                   xy=(i, 0.5), ha='center', va='center', fontsize=9)

    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Pass (1) / Fail (0)')
    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['FAIL', 'PASS'])

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_graph_with_correlations(positions: np.ndarray, adjacency: np.ndarray,
                                 C: np.ndarray, filepath: Path,
                                 title: str = 'Graph with Correlations',
                                 corr_threshold: float = 0.3):
    """
    Plot graph with edges colored by correlation strength.

    Args:
        positions: Node positions
        adjacency: Adjacency matrix
        C: Correlation matrix
        filepath: Output file path
        title: Plot title
        corr_threshold: Minimum correlation to show as edge
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 10))

    n = positions.shape[0]

    # Draw correlation edges (non-adjacent with high correlation)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 0 and C[i, j] > corr_threshold:
                alpha = min(C[i, j], 1.0)
                ax.plot([positions[i, 0], positions[j, 0]],
                       [positions[i, 1], positions[j, 1]],
                       'r-', alpha=alpha * 0.5, linewidth=0.5)

    # Draw graph edges
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > 0:
                ax.plot([positions[i, 0], positions[j, 0]],
                       [positions[i, 1], positions[j, 1]],
                       'b-', alpha=0.3, linewidth=1)

    # Draw nodes
    ax.scatter(positions[:, 0], positions[:, 1], s=50, c='blue', alpha=0.7, zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
