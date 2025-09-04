"""Graph utility functions for JraphX."""

from .coalesce import coalesce
from .convert import to_dense_adj, to_edge_index, to_undirected
from .degree import degree, in_degree, out_degree
from .loop import add_self_loops, remove_self_loops
from .num_nodes import maybe_num_nodes
from .scatter import (
    scatter,
    scatter_add,
    scatter_logsumexp,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_std,
)
from .scatter_softmax import masked_scatter_softmax, scatter_log_softmax, scatter_softmax

__all__ = [
    # Scatter operations
    "scatter",
    "scatter_add",
    "scatter_mean",
    "scatter_max",
    "scatter_min",
    "scatter_std",
    "scatter_logsumexp",
    # Scatter softmax operations
    "scatter_softmax",
    "scatter_log_softmax",
    "masked_scatter_softmax",
    # Degree utilities
    "degree",
    "in_degree",
    "out_degree",
    # Loop utilities
    "add_self_loops",
    "remove_self_loops",
    # Conversion utilities
    "to_undirected",
    "to_dense_adj",
    "to_edge_index",
    # Other utilities
    "maybe_num_nodes",
    "coalesce",
]
