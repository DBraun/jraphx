"""Edge coalescing utilities for removing duplicate edges."""

from jax import numpy as jnp

from jraphx.utils.scatter import scatter


def coalesce(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    num_nodes: int | None = None,
    reduce: str = "add",
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (jax.Array): The edge indices.
        edge_attr (jax.Array, optional): Edge weights
            or multi-dimensional edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. Used only to validate
            :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"` / :obj:`"sum"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`). (default: :obj:`"add"`)

    Returns:
        Tuple of (coalesced edge_index, coalesced edge_attr).

    Raises:
        ValueError: If :attr:`edge_index` addresses a node outside of
            :attr:`num_nodes`.

    .. note::
        The number of unique edges is data-dependent, so this function cannot
        be traced by :obj:`jax.jit`.
    """
    if edge_index.shape[1] == 0:
        return edge_index, edge_attr

    if num_nodes is not None:
        max_index = int(edge_index.max())
        if max_index >= num_nodes:
            raise ValueError(
                f"`edge_index` contains node {max_index}, which is out of range "
                f"for a graph with {num_nodes} nodes"
            )

    # Lexicographic de-duplication of the (row, col) columns. Working on the
    # index pair directly keeps this exact for any node count, unlike packing
    # the pair into a single integer key.
    unique_edge_index, inverse_indices = jnp.unique(edge_index, axis=1, return_inverse=True)
    inverse_indices = inverse_indices.reshape(-1)
    num_unique = unique_edge_index.shape[1]

    if edge_attr is not None:
        unique_attr = scatter(edge_attr, inverse_indices, dim_size=num_unique, dim=0, reduce=reduce)
        return unique_edge_index, unique_attr

    return unique_edge_index, None
