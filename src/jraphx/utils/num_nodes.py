"""Utilities for inferring the number of nodes in a graph."""

import jax


def maybe_num_nodes(
    edge_index: jax.Array,
    num_nodes: int | None = None,
) -> int:
    r"""Returns the number of nodes in the graph given by :attr:`edge_index`.

    Args:
        edge_index (jax.Array): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Returns:
        int: The number of nodes in the graph.

    Raises:
        jax.errors.ConcretizationTypeError: If :obj:`num_nodes` is :obj:`None`
            and :attr:`edge_index` is a tracer.

    .. note::
        Inferring the node count reads :attr:`edge_index` on the host, so it is
        not compatible with JIT compilation. Pass :obj:`num_nodes` explicitly
        inside jitted code.
    """
    if num_nodes is not None:
        return num_nodes

    if edge_index.size == 0:
        return 0

    return int(edge_index.max()) + 1
