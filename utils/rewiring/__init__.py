"""
Graph Rewiring Methods for MEMP

This module provides implementations of graph rewiring methods:
- SDRF (Stochastic Discrete Ricci Flow) - Curvature-based rewiring
- FoSR (First-order Spectral Rewiring) - Spectral gap optimization

Both methods can be applied to PyTorch Geometric Data objects.
"""

import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data

# Import rewiring implementations
# Use CPU version of SDRF by default (works without GPU)
from .sdrf_cpu import sdrf as sdrf_impl
from .fosr import edge_rewire as fosr_edge_rewire

# Try to import CUDA version if available (optional, for GPU acceleration)
try:
    from .cuda import sdrf as sdrf_cuda
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    sdrf_cuda = None


def apply_sdrf_rewiring(
    data: Union[Data, List[Data]],
    loops: int = 10,
    tau: float = 1.0,
    remove_edges: bool = False,
    removal_bound: float = 0.5,
    is_undirected: bool = True,
    use_cuda: bool = False
) -> Union[Data, List[Data]]:
    """
    Apply SDRF (Stochastic Discrete Ricci Flow) rewiring to graph(s).

    SDRF adds edges around negatively-curved edges to reduce bottlenecks
    and optionally removes high-curvature edges.

    Parameters:
    -----------
    data : Data or List[Data]
        PyG Data object(s) to rewire
    loops : int, default=10
        Number of rewiring iterations (each iteration adds one edge)
    tau : float, default=1.0
        Temperature parameter for edge addition sampling (higher = more random)
    remove_edges : bool, default=False
        Whether to remove high-curvature edges
    removal_bound : float, default=0.5
        Curvature threshold for edge removal (only used if remove_edges=True)
    is_undirected : bool, default=True
        Whether graph is undirected
    use_cuda : bool, default=False
        Whether to use CUDA GPU acceleration (requires GPU and CUDA-capable numba)

    Returns:
    --------
    Data or List[Data]
        Rewired graph(s) with modified edge_index

    Example:
    --------
    >>> from torch_geometric.data import Data
    >>> data = Data(x=..., edge_index=..., y=...)
    >>> rewired_data = apply_sdrf_rewiring(data, loops=10, tau=1.0)
    """
    # Choose implementation based on use_cuda parameter
    if use_cuda:
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA version requested but not available. Install CUDA-capable numba or set use_cuda=False.")
        sdrf_fn = sdrf_cuda
    else:
        sdrf_fn = sdrf_impl

    # Handle single graph vs list of graphs
    is_single = isinstance(data, Data)
    graphs = [data] if is_single else data

    rewired_graphs = []
    for graph in graphs:
        rewired = sdrf_fn(
            graph,
            loops=loops,
            remove_edges=remove_edges,
            removal_bound=removal_bound,
            tau=tau,
            is_undirected=is_undirected
        )
        rewired_graphs.append(rewired)

    return rewired_graphs[0] if is_single else rewired_graphs


def apply_fosr_rewiring(
    data: Union[Data, List[Data]],
    num_iterations: int = 50,
    initial_power_iters: int = 5
) -> Union[Data, List[Data]]:
    """
    Apply FoSR (First-order Spectral Rewiring) to graph(s).

    FoSR adds edges based on spectral expansion to improve graph connectivity
    and prevent oversquashing. Only adds edges (no removal).

    Parameters:
    -----------
    data : Data or List[Data]
        PyG Data object(s) to rewire
    num_iterations : int, default=50
        Number of edges to add (adds 2 directed edges per iteration for undirected)
    initial_power_iters : int, default=5
        Number of power iterations to find the initial eigenvector

    Returns:
    --------
    Data or List[Data]
        Rewired graph(s) with modified edge_index and added edge_type attribute
        edge_type: 0 = original edge, 1 = added edge

    Example:
    --------
    >>> from torch_geometric.data import Data
    >>> data = Data(x=..., edge_index=..., y=...)
    >>> rewired_data = apply_fosr_rewiring(data, num_iterations=50)
    >>> print(rewired_data.edge_type)  # 0 for original, 1 for added edges
    """
    # Handle single graph vs list of graphs
    is_single = isinstance(data, Data)
    graphs = [data] if is_single else data

    rewired_graphs = []
    for graph in graphs:
        # Convert PyG edge_index to numpy
        edge_index_np = graph.edge_index.cpu().numpy()

        # Apply FoSR rewiring
        edge_index_new, edge_type, _ = fosr_edge_rewire(
            edge_index_np,
            num_iterations=num_iterations,
            initial_power_iters=initial_power_iters
        )

        # Create new Data object with rewired edges
        rewired = graph.clone()
        rewired.edge_index = torch.tensor(edge_index_new, dtype=torch.long)
        rewired.edge_type = torch.tensor(edge_type, dtype=torch.long)

        # Move to same device as original
        if graph.edge_index.is_cuda:
            rewired.edge_index = rewired.edge_index.cuda()
            rewired.edge_type = rewired.edge_type.cuda()

        rewired_graphs.append(rewired)

    return rewired_graphs[0] if is_single else rewired_graphs


def apply_rewiring(
    data: Union[Data, List[Data]],
    method: str,
    **kwargs
) -> Union[Data, List[Data]]:
    """
    Unified interface for applying graph rewiring.

    Parameters:
    -----------
    data : Data or List[Data]
        PyG Data object(s) to rewire
    method : str
        Rewiring method: 'sdrf' or 'fosr'
    **kwargs : dict
        Method-specific parameters

    Returns:
    --------
    Data or List[Data]
        Rewired graph(s)

    Raises:
    -------
    ValueError
        If method is not 'sdrf' or 'fosr'

    Example:
    --------
    >>> # SDRF rewiring
    >>> rewired = apply_rewiring(data, method='sdrf', loops=10, tau=1.0)
    >>>
    >>> # FoSR rewiring
    >>> rewired = apply_rewiring(data, method='fosr', num_iterations=50)
    """
    if method == 'sdrf':
        return apply_sdrf_rewiring(data, **kwargs)
    elif method == 'fosr':
        return apply_fosr_rewiring(data, **kwargs)
    else:
        raise ValueError(f"Unknown rewiring method: {method}. Choose 'sdrf' or 'fosr'.")


def compute_rewiring_stats(
    original: Union[Data, List[Data]],
    rewired: Union[Data, List[Data]]
) -> dict:
    """
    Compute statistics about graph rewiring (edge additions/removals).

    Parameters:
    -----------
    original : Data or List[Data]
        Original graph(s) before rewiring
    rewired : Data or List[Data]
        Rewired graph(s)

    Returns:
    --------
    dict
        Statistics including:
        - avg_edges_original: Average number of edges in original graphs
        - avg_edges_rewired: Average number of edges after rewiring
        - avg_edges_added: Average number of edges added
        - pct_edges_added: Percentage of edges added relative to original

    Example:
    --------
    >>> stats = compute_rewiring_stats(original_graphs, rewired_graphs)
    >>> print(f"Added {stats['avg_edges_added']:.1f} edges on average")
    """
    # Handle single graph vs list of graphs
    original_list = [original] if isinstance(original, Data) else original
    rewired_list = [rewired] if isinstance(rewired, Data) else rewired

    assert len(original_list) == len(rewired_list), "Mismatched number of graphs"

    stats = {
        'avg_edges_original': 0,
        'avg_edges_rewired': 0,
        'avg_edges_added': 0,
        'pct_edges_added': 0
    }

    for orig, rew in zip(original_list, rewired_list):
        # Count undirected edges (divide by 2 if edges are bidirectional)
        orig_edges = orig.edge_index.shape[1] // 2
        rew_edges = rew.edge_index.shape[1] // 2

        stats['avg_edges_original'] += orig_edges
        stats['avg_edges_rewired'] += rew_edges
        stats['avg_edges_added'] += (rew_edges - orig_edges)

    n_graphs = len(original_list)
    for key in stats:
        stats[key] /= n_graphs

    if stats['avg_edges_original'] > 0:
        stats['pct_edges_added'] = (stats['avg_edges_added'] / stats['avg_edges_original']) * 100

    return stats


__all__ = [
    'apply_sdrf_rewiring',
    'apply_fosr_rewiring',
    'apply_rewiring',
    'compute_rewiring_stats',
    'sdrf_impl',
    'sdrf_cuda',
    'fosr_edge_rewire',
    'CUDA_AVAILABLE'
]
