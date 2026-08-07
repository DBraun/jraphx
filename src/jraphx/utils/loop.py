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
    In case the graph is weighted or has multi-dimensional edge features
    (:attr:`edge_attr` is not :obj:`None`), edge features of self-loops will be
    added according to :obj:`fill_value`. One self-loop is appended per node
    unconditionally, so a node that already has a self-loop ends up with two;
    use :func:`add_remaining_self_loops` to add only the missing ones.

    Args:
        edge_index (jax.Array): The edge indices.
        edge_attr (jax.Array, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        fill_value (float or str, optional): The way to generate edge features of
            self-loops. If float, edge features are set to this value.
            If str, edge features are computed by aggregating existing edge features
            that point to each node using the specified reduction ('mean', 'add'
            (alias 'sum'), 'max', 'min'). (default: :obj:`1.0`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Returns:
        Tuple of (edge_index with self-loops, edge_attr with self-loops).

    .. note::
        The output shape depends on :obj:`num_nodes`, so it must be given as a
        static integer under :obj:`jax.jit`. Inferring it from
        :attr:`edge_index` reads the array on the host.
    """
    if num_nodes is None:
        if edge_index.size == 0:
            return edge_index, edge_attr
        num_nodes = int(edge_index.max()) + 1

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

    # One self-loop per node
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
    """Add self-loops so that every node carries exactly one.

    The self-loops the input already had are removed first, so a duplicated
    loop collapses to a single one. Every node's loop is appended at the end of
    the edge list in node order. A node that already had a loop keeps that
    loop's attribute (the last occurrence, if it had several); loops created
    for the remaining nodes take :obj:`fill_value`.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_attr: Optional edge attributes [num_edges, \\*]
        fill_value: Value to use for self-loop edge attributes
        num_nodes: Number of nodes

    Returns:
        Tuple of (edge_index with self-loops, edge_attr with self-loops)

    .. note::
        The number of self-loops removed from the input is data-dependent, so
        this function cannot be traced by :obj:`jax.jit`.
    """
    if num_nodes is None:
        if edge_index.size == 0:
            return edge_index, edge_attr
        num_nodes = int(edge_index.max()) + 1

    # Drop every existing self-loop; the per-node loops appended below replace
    # them, so a duplicated loop collapses to one
    row, col = edge_index[0], edge_index[1]
    keep = row != col

    loop_index = jnp.arange(num_nodes, dtype=edge_index.dtype)
    edge_index = jnp.concatenate(
        [edge_index[:, keep], jnp.stack([loop_index, loop_index], axis=0)], axis=1
    )

    if edge_attr is not None:
        loop_attr = jnp.full(
            (num_nodes,) + edge_attr.shape[1:],
            fill_value,
            dtype=edge_attr.dtype,
        )
        # A node that already had a loop keeps its attribute; with several,
        # the last occurrence wins
        loop_attr = loop_attr.at[row[~keep]].set(edge_attr[~keep])
        edge_attr = jnp.concatenate([edge_attr[keep], loop_attr], axis=0)

    return edge_index, edge_attr
