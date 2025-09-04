"""Self-loop utilities for graphs."""

from typing import Union

from jax import numpy as jnp

from jraphx.utils.scatter import scatter


def add_self_loops(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    fill_value: Union[float, str] = 1.0,
    num_nodes: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (jax.Array): The edge indices.
        edge_attr (jax.Array, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        fill_value (float or str, optional): The way to generate edge features of
            self-loops. If float, edge features are set to this value.
            If str, edge features are computed by aggregating existing edge features
            that point to each node using the specified reduction ('mean', 'add', 'max', 'min').
            (default: :obj:`1.0`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Returns:
        Tuple of (edge_index with self-loops, edge_attr with self-loops).

    .. note::
        For JIT compatibility, :obj:`num_nodes` should be provided as a static
        integer when possible.
    """
    # JIT-compatible: Use shape directly when num_nodes not provided
    if num_nodes is None:
        if edge_index.size == 0:
            return edge_index, edge_attr
        num_nodes = edge_index.max() + 1

    # Handle edge attributes first (before modifying edge_index)
    if edge_attr is not None:
        if isinstance(fill_value, str):
            # Use scatter to compute aggregated features for self-loops using original edges
            target_nodes = edge_index[1]  # Target nodes of existing edges
            loop_attr = scatter(
                edge_attr, target_nodes, dim_size=num_nodes, dim=0, reduce=fill_value
            )
        else:
            # Create self-loop attributes with constant value
            if edge_attr.ndim == 1:
                loop_attr = jnp.full(num_nodes, fill_value, dtype=edge_attr.dtype)
            else:
                loop_attr = jnp.full(
                    (num_nodes,) + edge_attr.shape[1:],
                    fill_value,
                    dtype=edge_attr.dtype,
                )
        edge_attr = jnp.concatenate([edge_attr, loop_attr], axis=0)

    # Create self-loop edges - using dynamic shape-dependent arange
    # This works in JIT because num_nodes is now a traced value
    loop_index = jnp.arange(num_nodes)
    loop_index = jnp.stack([loop_index, loop_index], axis=0)

    # Concatenate with existing edges
    edge_index = jnp.concatenate([edge_index, loop_index], axis=1)

    return edge_index, edge_attr


def remove_self_loops(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Remove self-loops from edge indices.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_attr: Optional edge attributes [num_edges, \\*]

    Returns:
        Tuple of (edge_index without self-loops, edge_attr without self-loops)
    """
    # Find non-self-loop edges
    mask = edge_index[0] != edge_index[1]

    # Filter edges
    edge_index = edge_index[:, mask]

    # Filter edge attributes if present
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return edge_index, edge_attr


def add_remaining_self_loops(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    fill_value: float = 1.0,
    num_nodes: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Add self-loops only for nodes that don't already have them.

    Optimized version using boolean masking instead of setdiff1d.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_attr: Optional edge attributes [num_edges, \\*]
        fill_value: Value to use for self-loop edge attributes
        num_nodes: Number of nodes

    Returns:
        Tuple of (edge_index with self-loops, edge_attr with self-loops)
    """
    # JIT-compatible: Use shape directly when num_nodes not provided
    if num_nodes is None:
        if edge_index.size == 0:
            return edge_index, edge_attr
        num_nodes = edge_index.max() + 1

    # Find existing self-loops
    row, col = edge_index[0], edge_index[1]
    is_self_loop = row == col

    # Create a boolean mask for nodes that have self-loops
    # More efficient than using unique and setdiff1d
    has_self_loop = jnp.zeros(num_nodes, dtype=jnp.bool_)
    has_self_loop = has_self_loop.at[row[is_self_loop]].set(True)

    # Find nodes without self-loops using boolean indexing
    nodes_without_loops = jnp.where(~has_self_loop)[0]

    # Create self-loops for nodes without them
    num_new_loops = nodes_without_loops.size
    if num_new_loops > 0:
        loop_index = jnp.stack([nodes_without_loops, nodes_without_loops], axis=0)
        edge_index = jnp.concatenate([edge_index, loop_index], axis=1)

        # Handle edge attributes
        if edge_attr is not None:
            if edge_attr.ndim == 1:
                loop_attr = jnp.full(
                    num_new_loops,
                    fill_value,
                    dtype=edge_attr.dtype,
                )
            else:
                loop_attr = jnp.full(
                    (num_new_loops,) + edge_attr.shape[1:],
                    fill_value,
                    dtype=edge_attr.dtype,
                )
            edge_attr = jnp.concatenate([edge_attr, loop_attr], axis=0)

    return edge_index, edge_attr
