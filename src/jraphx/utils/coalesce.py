"""Edge coalescing utilities for removing duplicate edges."""

from jax import numpy as jnp

from jraphx.utils.scatter import scatter_add, scatter_max, scatter_mean, scatter_min


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
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"add"`)

    Returns:
        Tuple of (coalesced edge_index, coalesced edge_attr).

    .. note::
        For JIT compatibility, :obj:`num_nodes` should be provided as a static
        integer when possible.
    """
    if edge_index.shape[1] == 0:
        return edge_index, edge_attr

    # Create unique edge identifiers
    row, col = edge_index[0], edge_index[1]

    # Get maximum node index to create unique IDs
    if num_nodes is None:
        num_nodes = jnp.max(edge_index) + 1

    # Create unique edge IDs by combining row and col indices
    edge_ids = row * num_nodes + col

    # Find unique edges
    unique_ids, inverse_indices = jnp.unique(edge_ids, return_inverse=True)

    # Reconstruct unique edge indices
    unique_row = unique_ids // num_nodes
    unique_col = unique_ids % num_nodes
    unique_edge_index = jnp.stack([unique_row, unique_col], axis=0)

    # Handle edge attributes
    if edge_attr is not None:
        # Aggregate duplicate edge attributes
        if reduce == "add":
            unique_attr = scatter_add(edge_attr, inverse_indices, dim_size=len(unique_ids), dim=0)
        elif reduce == "mean":
            unique_attr = scatter_mean(edge_attr, inverse_indices, dim_size=len(unique_ids), dim=0)
        elif reduce == "max":
            unique_attr = scatter_max(edge_attr, inverse_indices, dim_size=len(unique_ids), dim=0)
        elif reduce == "min":
            unique_attr = scatter_min(edge_attr, inverse_indices, dim_size=len(unique_ids), dim=0)
        else:
            raise ValueError(f"Unknown reduce operation: {reduce}")

        return unique_edge_index, unique_attr

    return unique_edge_index, None
