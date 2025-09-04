"""Utilities for inferring the number of nodes in a graph."""

from jax import numpy as jnp


def maybe_num_nodes(
    edge_index: jnp.ndarray,
    num_nodes: int | None = None,
) -> int:
    r"""Returns the number of nodes in the graph given by :attr:`edge_index`.

    Args:
        edge_index (jax.Array): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Returns:
        int: The number of nodes in the graph.

    .. note::
        This function may not be compatible with JIT compilation when
        :obj:`num_nodes` is :obj:`None`, as it requires computing the
        maximum value of :attr:`edge_index` at runtime.
    """
    if num_nodes is not None:
        return num_nodes

    if edge_index.size == 0:
        return 0

    return edge_index.max() + 1
