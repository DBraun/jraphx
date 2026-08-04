"""Graph conversion utilities."""

from jax import numpy as jnp

from jraphx.utils.coalesce import coalesce
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

    The result is coalesced: the edge list is row-wise sorted, duplicated
    edges appear once, and their features are merged with :attr:`reduce`.

    Args:
        edge_index (jax.Array): The edge indices.
        edge_attr (jax.Array, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"` / :obj:`"sum"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`). (default: :obj:`"add"`)

    Returns:
        Tuple of (undirected edge_index, undirected edge_attr).

    .. note::
        The number of unique edges is data-dependent, so this function cannot
        be traced by :obj:`jax.jit`.
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

    return coalesce(edge_index, edge_attr, num_nodes, reduce=reduce)


def to_dense_adj(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    max_num_nodes: int | None = None,
) -> jnp.ndarray:
    """Convert edge indices to dense adjacency matrix.

    Parallel edges are accumulated, so a duplicated edge contributes the sum of
    its attributes (or its multiplicity when :attr:`edge_attr` is :obj:`None`).

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_attr: Optional edge attributes [num_edges] or [num_edges, num_features]
        max_num_nodes: Number of nodes of the dense output. Inferred from
            ``edge_index`` when :obj:`None`, which requires a concrete
            ``edge_index``.

    Returns:
        Dense adjacency matrix [num_nodes, num_nodes] or [num_nodes, num_nodes, num_features]
    """
    num_nodes = maybe_num_nodes(edge_index, max_num_nodes)

    if edge_attr is None:
        # Binary adjacency matrix
        adj = jnp.zeros((num_nodes, num_nodes), dtype=jnp.float32)
        return adj.at[edge_index[0], edge_index[1]].add(1.0)

    if edge_attr.ndim == 1:
        # Weighted adjacency matrix
        adj = jnp.zeros((num_nodes, num_nodes), dtype=edge_attr.dtype)
    else:
        # Multi-feature adjacency tensor
        num_features = edge_attr.shape[-1]
        adj = jnp.zeros((num_nodes, num_nodes, num_features), dtype=edge_attr.dtype)

    return adj.at[edge_index[0], edge_index[1]].add(edge_attr)


def to_edge_index(adj: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert adjacency matrix to edge indices.

    An edge is emitted for every entry that is non-zero (for a feature tensor,
    for every entry with at least one non-zero feature), and its stored value is
    always returned as the edge attribute.

    Args:
        adj: Adjacency matrix [num_nodes, num_nodes] or [num_nodes, num_nodes, num_features]

    Returns:
        Tuple of (edge_index [2, num_edges], edge_attr [num_edges] or [num_edges, num_features])

    .. note::
        The number of edges is data-dependent, so this function cannot be
        traced by :obj:`jax.jit`.
    """
    if adj.ndim == 2:
        # Binary or weighted adjacency matrix
        row, col = jnp.where(adj != 0)
    else:
        # Multi-feature adjacency tensor: keep entries with any non-zero feature
        row, col = jnp.where(jnp.any(adj != 0, axis=-1))

    edge_index = jnp.stack([row, col], axis=0)
    edge_attr = adj[row, col]

    return edge_index, edge_attr
