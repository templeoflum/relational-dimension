"""
Graph generation module for Experiment 01.

Provides functions to create random geometric graphs and lattice graphs,
and extract their properties (positions, adjacency, geodesic distances).
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from typing import Tuple, Optional


def create_rgg(n: int, radius: float, seed: Optional[int] = None) -> nx.Graph:
    """
    Create a random geometric graph with N nodes.

    Nodes are placed uniformly at random in [0,1]^2, and edges connect
    pairs of nodes within the specified radius.

    Args:
        n: Number of nodes
        radius: Connection radius (nodes within this distance are connected)
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph with 'pos' node attributes
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random positions in unit square
    positions = np.random.rand(n, 2)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Store positions as node attributes
    for i in range(n):
        G.nodes[i]['pos'] = positions[i]

    # Compute pairwise distances and add edges
    distances = cdist(positions, positions)
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < radius:
                G.add_edge(i, j)

    return G


def create_lattice(side: int) -> nx.Graph:
    """
    Create a 2D square lattice graph (side x side).

    Args:
        side: Side length of the lattice

    Returns:
        NetworkX graph with 'pos' node attributes (grid positions)
    """
    # Create 2D grid graph
    G = nx.grid_2d_graph(side, side)

    # Convert node labels from (i,j) tuples to integers
    # and store positions
    mapping = {}
    for idx, (i, j) in enumerate(sorted(G.nodes())):
        mapping[(i, j)] = idx

    G = nx.relabel_nodes(G, mapping)

    # Store normalized positions [0,1]^2
    for node in G.nodes():
        i = node // side
        j = node % side
        # Normalize to [0,1] range
        G.nodes[node]['pos'] = np.array([i / (side - 1), j / (side - 1)])

    return G


def get_positions(graph: nx.Graph) -> np.ndarray:
    """
    Extract node positions from graph.

    Args:
        graph: NetworkX graph with 'pos' node attributes

    Returns:
        N x 2 array of node positions
    """
    n = graph.number_of_nodes()
    positions = np.zeros((n, 2))
    for i in range(n):
        positions[i] = graph.nodes[i]['pos']
    return positions


def get_adjacency(graph: nx.Graph) -> np.ndarray:
    """
    Get adjacency matrix from graph.

    Args:
        graph: NetworkX graph

    Returns:
        N x N adjacency matrix (symmetric, 0/1 entries)
    """
    return nx.to_numpy_array(graph)


def get_graph_distances(graph: nx.Graph) -> np.ndarray:
    """
    Compute all-pairs shortest path distances.

    Args:
        graph: NetworkX graph

    Returns:
        N x N distance matrix (inf for disconnected pairs)
    """
    n = graph.number_of_nodes()

    # Handle disconnected graphs by using largest connected component
    if not nx.is_connected(graph):
        # Get largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc).copy()
        # Relabel nodes to 0..len(largest_cc)-1
        mapping = {old: new for new, old in enumerate(sorted(largest_cc))}
        subgraph = nx.relabel_nodes(subgraph, mapping)
        graph = subgraph
        n = graph.number_of_nodes()

    # Compute all-pairs shortest paths
    path_lengths = dict(nx.all_pairs_shortest_path_length(graph))

    # Convert to matrix
    D = np.full((n, n), np.inf)
    for i in range(n):
        for j, dist in path_lengths[i].items():
            D[i, j] = dist

    return D


def get_largest_component(graph: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component from a graph.

    Args:
        graph: NetworkX graph

    Returns:
        Subgraph containing only the largest connected component,
        with nodes relabeled to 0..n-1
    """
    if nx.is_connected(graph):
        return graph

    # Get largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc).copy()

    # Relabel nodes to 0..len(largest_cc)-1
    old_to_new = {old: new for new, old in enumerate(sorted(largest_cc))}
    subgraph = nx.relabel_nodes(subgraph, old_to_new)

    return subgraph


def estimate_radius(n: int, target_degree: float = 6.0) -> float:
    """
    Estimate radius for RGG to achieve target average degree.

    For RGG in unit square, expected degree ≈ n * pi * r^2.

    Args:
        n: Number of nodes
        target_degree: Desired average degree

    Returns:
        Estimated radius
    """
    # E[degree] = n * pi * r^2 - 1 (excluding self)
    # r = sqrt((target_degree + 1) / (n * pi))
    return np.sqrt((target_degree + 1) / (n * np.pi))
