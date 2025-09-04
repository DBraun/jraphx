"""Graph conversion utilities."""

from jax import numpy as jnp

from jraphx.utils.num_nodes import maybe_num_nodes


def to_undirected(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    num_nodes: int | None = None,
    reduce: str = "add",
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (jax.Array): The edge indices.
        edge_attr (jax.Array, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"add"`)

    Returns:
        Tuple of (undirected edge_index, undirected edge_attr).
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # Add reverse edges
    row, col = edge_index[0], edge_index[1]
    row_rev, col_rev = col, row

    # Concatenate forward and reverse edges
    edge_index = jnp.concatenate(
        [edge_index, jnp.stack([row_rev, col_rev], axis=0)],
        axis=1,
    )

    # Handle edge attributes
    if edge_attr is not None:
        edge_attr = jnp.concatenate([edge_attr, edge_attr], axis=0)

    # Remove duplicates (will be implemented in coalesce)
    # For now, just return the concatenated version
    return edge_index, edge_attr


def to_dense_adj(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    max_num_nodes: int | None = None,
) -> jnp.ndarray:
    """Convert edge indices to dense adjacency matrix.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_attr: Optional edge attributes [num_edges] or [num_edges, num_features]
        max_num_nodes: Maximum number of nodes (for padding)

    Returns:
        Dense adjacency matrix [num_nodes, num_nodes] or [num_nodes, num_nodes, num_features]
    """
    num_nodes = maybe_num_nodes(edge_index, max_num_nodes)

    if edge_attr is None:
        # Binary adjacency matrix
        adj = jnp.zeros((num_nodes, num_nodes), dtype=jnp.float32)
        adj = adj.at[edge_index[0], edge_index[1]].set(1.0)
    else:
        if edge_attr.ndim == 1:
            # Weighted adjacency matrix
            adj = jnp.zeros((num_nodes, num_nodes), dtype=edge_attr.dtype)
            adj = adj.at[edge_index[0], edge_index[1]].set(edge_attr)
        else:
            # Multi-feature adjacency tensor
            num_features = edge_attr.shape[-1]
            adj = jnp.zeros((num_nodes, num_nodes, num_features), dtype=edge_attr.dtype)
            for i in range(edge_attr.shape[0]):
                adj = adj.at[edge_index[0, i], edge_index[1, i]].set(edge_attr[i])

    return adj


def to_edge_index(adj: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Convert adjacency matrix to edge indices.

    Args:
        adj: Adjacency matrix [num_nodes, num_nodes] or [num_nodes, num_nodes, num_features]

    Returns:
        Tuple of (edge_index [2, num_edges], edge_attr [num_edges] or [num_edges, num_features])
    """
    if adj.ndim == 2:
        # Binary or weighted adjacency matrix
        row, col = jnp.where(adj != 0)
        edge_index = jnp.stack([row, col], axis=0)

        # Get edge weights if not binary
        edge_attr = adj[row, col]
        if jnp.all(edge_attr == 1):
            edge_attr = None
    else:
        # Multi-feature adjacency tensor
        # Check if any feature is non-zero
        mask = jnp.any(adj != 0, axis=-1)
        row, col = jnp.where(mask)
        edge_index = jnp.stack([row, col], axis=0)
        edge_attr = adj[row, col]

    return edge_index, edge_attr
