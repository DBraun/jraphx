"""Optimized degree computation utilities for graphs using JAX operations."""

import jax
from jax import numpy as jnp

from jraphx.utils.num_nodes import maybe_num_nodes


def degree(
    index: jax.Array,
    num_nodes: int | None = None,
    dtype: jnp.dtype | None = None,
) -> jax.Array:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (jax.Array): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.* the maximum
            entry of ``index`` plus one. (default: :obj:`None`)
        dtype (jnp.dtype, optional): The desired data type of the
            returned tensor.

    Returns:
        Node degrees, one entry per node.

    Example:
        >>> import jax.numpy as jnp
        >>> row = jnp.array([0, 1, 0, 2, 0])
        >>> degree(row, dtype=jnp.int32)
        Array([3, 1, 1], dtype=int32)
    """
    num_nodes = maybe_num_nodes(index.reshape(1, -1), num_nodes)
    dtype = dtype or jnp.float32

    # Direct use of segment_sum is more efficient than scatter_add with ones
    return jax.ops.segment_sum(
        jnp.ones(index.shape[0], dtype=dtype),
        index,
        num_segments=num_nodes,
    )


def in_degree(
    edge_index: jax.Array,
    num_nodes: int | None = None,
    dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Compute the in-degree of nodes.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        dtype: Output data type

    Returns:
        Node in-degrees [num_nodes]
    """
    # Size the result by every node the graph mentions: a node appearing only
    # as a source has in-degree zero, not a missing row.
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    return degree(edge_index[1], num_nodes, dtype)


def out_degree(
    edge_index: jax.Array,
    num_nodes: int | None = None,
    dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Compute the out-degree of nodes.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        dtype: Output data type

    Returns:
        Node out-degrees [num_nodes]
    """
    # Size the result by every node the graph mentions: a node appearing only
    # as a target has out-degree zero, not a missing row.
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    return degree(edge_index[0], num_nodes, dtype)
