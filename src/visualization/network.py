import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colormaps

from .base import HEIGHT, WIDTH, save_figure, truncate_colormap


def build_graph(
    matrix: np.ndarray,
    node_labels: list[str] | None = None,
    threshold: float = 1e-4,
    directed: bool = False,
) -> nx.Graph:
    """
    Construct a NetworkX graph from a square matrix (e.g., precision, covariance, or
    correlation matrix).

    Use this to visualize:
    - Conditional dependencies (precision matrix)
    - Marginal dependencies (covariance or correlation matrix)
    - Network structures inferred from asset relationships

    Parameters
    ----------
    matrix : np.ndarray
        A square (N x N) matrix representing relationships between assets.
        Common examples include precision (inverse covariance), covariance, or
        correlation matrices.
    node_labels : list of str, optional
        List of node labels corresponding to the rows/columns of the matrix.
        If None, defaults to generic names A0, A1, ..., AN.
    threshold : float
        Minimum absolute value of an off-diagonal entry to include an edge between
        nodes.
    directed : bool
        If True, builds a directed graph. Otherwise, builds an undirected graph.

    Returns
    -------
    G : networkx.Graph or networkx.DiGraph
        The constructed graph with nodes as assets and edges weighted by matrix values.
    """
    n = matrix.shape[0]
    if node_labels is None:
        node_labels = [f"A{i}" for i in range(n)]

    G = nx.DiGraph() if directed else nx.Graph()

    for i in range(n):
        G.add_node(node_labels[i])

    for i in range(n):
        for j in range(i + 1 if not directed else 0, n):
            if i == j:
                continue
            weight = matrix[i, j]
            if np.abs(weight) > threshold:
                G.add_edge(node_labels[i], node_labels[j], weight=weight)

    return G


def plot_graph(
    G: nx.Graph,
    layout: str = "circular",
    figsize: tuple = (WIDTH, HEIGHT),
    title: str = "Graph Structure",
    node_color: str = "tab:blue",
    edge_cmap: str = "Blues",
    edge_vmin: float | None = None,
    edge_vmax: float | None = None,
    saveas_filename: str | None = None,
    **kwargs,
) -> None:
    """
    Plot a generic weighted graph using NetworkX and matplotlib.

    Parameters
    ----------
    G : networkx.Graph
        Graph object with nodes and weighted edges.
    layout : {'spring', 'kamada_kawai', 'circular'}, default='circular'
        Graph layout algorithm for node placement.
    figsize : tuple, default=(10, 8)
        Size of the figure in inches.
    title : str, default='Graph Structure'
        Title of the plot.
    node_color : str, default='tab:blue'
        Color for the graph nodes.
    edge_cmap : str, default='Blues'
        Colormap to apply to edge weights (if edges are weighted).
    edge_vmin : float, optional
        Minimum value for edge colormap normalization (if edges are weighted).
    edge_vmax : float, optional
        Maximum value for edge colormap normalization (if edges are weighted).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Node layout.
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)

    # Draw nodes and labels.
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="lightgray", font_size=10)

    # Attempt to extract and draw weighted edges, but proceed even if none are found.
    edge_attr = nx.get_edge_attributes(G, "weight")
    if edge_attr:
        edges, weights = zip(*edge_attr.items())

        # Ensure weights is a list of floats for plotting
        # This also helps with potential type issues for edge_color
        weights_list = [float(w) for w in weights]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=weights_list,  # type: ignore[arg-type]
            edge_cmap=truncate_colormap(colormaps[edge_cmap], minval=0.01, maxval=0.9),
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            width=2,
            ax=ax,
        )
    else:
        # If no weighted edges, draw unweighted edges with a default color (gray).
        nx.draw_networkx_edges(
            G,
            pos,
            width=2,
            edge_color="gray",  # Default color for unweighted edges.
            ax=ax,
        )
        print("Warning: Graph has no weighted edges. Drawing unweighted edges in gray.")

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, saveas_filename or "graph", **kwargs)
