"""Base message passing class for graph neural networks in JAX/NNX.

This module provides an optimized base class for message passing operations
using JAX's efficient indexing and gathering operations.
"""

from typing import Any, Literal, Union

import jax.numpy as jnp
from flax.nnx import Module
from jax.core import Tracer

from jraphx.utils.scatter import scatter_add, scatter_max, scatter_mean, scatter_min


def _validate_index_range(index: jnp.ndarray, num_nodes: int, role: str) -> None:
    """Check that gather indices address an existing row of a node table.

    :func:`jax.numpy.take` fills out-of-range positions with ``NaN`` instead of
    raising, which silently poisons the aggregated output. The check is skipped
    for traced indices, since their values are unavailable at trace time.

    Args:
        index: Node indices used to gather one endpoint of every edge.
        num_nodes: Number of rows of the table being gathered from.
        role: Endpoint name used in the error message.

    Raises:
        IndexError: If a concrete index is negative or not smaller than
            ``num_nodes``.
    """
    if isinstance(index, Tracer):
        return

    index = jnp.asarray(index)
    if index.size == 0:
        return

    lowest = int(index.min())
    highest = int(index.max())
    if lowest < 0 or highest >= num_nodes:
        raise IndexError(
            f"{role} indices span [{lowest}, {highest}], which does not fit a node "
            f"table of {num_nodes} rows"
        )


class MessagePassing(Module):
    r"""Base class for creating message passing layers.

    Message passing layers follow the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\bigoplus` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean, min, max or mul, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
    differentiable functions such as MLPs.

    Args:
        aggr (str, optional): The aggregation scheme to use, *e.g.*,
            :obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`.
            (default: :obj:`"add"`)
        flow (str, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
    """

    def __init__(
        self,
        aggr: str = "add",
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
        node_dim: int = -2,
    ):
        """Initialize the message passing layer.

        Args:
            aggr: Aggregation method for messages
            flow: Direction of message flow
            node_dim: Dimension for node features
        """
        self.aggr = aggr
        self.flow = flow
        self.node_dim = node_dim

        # Validate inputs
        if aggr not in ["add", "mean", "max", "min"]:
            raise ValueError(f"Unknown aggregation: {aggr}")
        if flow not in ["source_to_target", "target_to_source"]:
            raise ValueError(f"Unknown flow: {flow}")

    def propagate(
        self,
        edge_index: jnp.ndarray,
        x: Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]],
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        """Main propagation step that orchestrates message passing.

        Messages travel from source nodes :math:`j` to target nodes :math:`i`.
        A bipartite graph is expressed by passing ``x`` as a tuple
        ``(x_src, x_dst)``: ``x[0]`` holds the features of the source set and
        ``x[1]`` those of the target set. The result therefore has
        ``x[1].shape[0]`` rows for :obj:`flow="source_to_target"` and
        ``x[0].shape[0]`` rows for :obj:`flow="target_to_source"`.

        Args:
            edge_index: Edge indices [2, num_edges]
            x: Node features [num_nodes, features], or a ``(x_src, x_dst)`` tuple
                for bipartite graphs
            edge_attr: Optional edge features [num_edges, edge_features]
            size: Optional size (num_src_nodes, num_dst_nodes) for bipartite graphs

        Returns:
            Updated node features after message passing

        Raises:
            ValueError: If ``x`` is a tuple that does not hold exactly two feature
                tables, or if such a tuple disagrees with an explicit ``size``.
            IndexError: If a bipartite edge addresses a node outside of its table.
        """
        # The roles of the two rows of ``edge_index`` and of the two node tables
        # swap together with the flow direction.
        if self.flow == "source_to_target":
            row, col = edge_index[0], edge_index[1]
            src_pos, dst_pos = 0, 1
        else:
            row, col = edge_index[1], edge_index[0]
            src_pos, dst_pos = 1, 0

        if isinstance(x, tuple):
            if len(x) != 2:
                raise ValueError(
                    f"Bipartite `x` must be a 2-tuple (x_src, x_dst), got {len(x)} entries"
                )
            table_sizes = (x[0].shape[0], x[1].shape[0])
            if size is None:
                size = table_sizes
            elif tuple(size) != table_sizes:
                raise ValueError(
                    f"size={tuple(size)} disagrees with the bipartite feature tables "
                    f"{table_sizes}"
                )
            x_src_table, x_dst_table = x[src_pos], x[dst_pos]
            _validate_index_range(row, size[src_pos], "Source")
            _validate_index_range(col, size[dst_pos], "Target")
            x_original = x[dst_pos]
        else:
            x_src_table = x_dst_table = x
            # If size is explicitly provided, use it (for bipartite cases)
            if size is None:
                size = (x.shape[0], x.shape[0])
            x_original = x

        # Number of target nodes the messages are scattered into
        dim_size = size[dst_pos]

        # A subclass may fuse message construction and aggregation; the base class
        # only provides the hook, so dispatch when it is genuinely overridden.
        if type(self).message_and_aggregate is not MessagePassing.message_and_aggregate:
            aggr_out = self.message_and_aggregate(x, edge_index, edge_attr, dim_size)
        else:
            # Use efficient JAX indexing for gathering node features
            # jnp.take is more efficient than direct indexing for large arrays
            x_j_gathered = jnp.take(x_src_table, row, axis=0)  # Source nodes
            x_i_gathered = jnp.take(x_dst_table, col, axis=0)  # Target nodes
            messages = self.message(x_j_gathered, x_i_gathered, edge_attr)
            aggr_out = self.aggregate(messages, col, dim_size)

        out = self.update(aggr_out, x_original)

        return out

    def message(
        self,
        x_j: jnp.ndarray,
        x_i: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Construct messages from source nodes j to target nodes i.

        Args:
            x_j: Source node features [num_edges, features]
            x_i: Target node features [num_edges, features]
            edge_attr: Optional edge features [num_edges, edge_features]

        Returns:
            Messages [num_edges, message_features]
        """
        # Default: just return source features
        return x_j

    def aggregate(
        self,
        messages: jnp.ndarray,
        index: jnp.ndarray,
        dim_size: int | None = None,
    ) -> jnp.ndarray:
        """Aggregate messages at target nodes using optimized scatter operations.

        Args:
            messages: Messages to aggregate [num_edges, features]
            index: Target node indices [num_edges]
            dim_size: Number of target nodes

        Returns:
            Aggregated messages [num_nodes, features]
        """
        # Use optimized scatter operations (already using JAX segment ops)
        if self.aggr == "add":
            return scatter_add(messages, index, dim_size, dim=0)
        elif self.aggr == "mean":
            return scatter_mean(messages, index, dim_size, dim=0)
        elif self.aggr == "max":
            return scatter_max(messages, index, dim_size, dim=0)
        elif self.aggr == "min":
            return scatter_min(messages, index, dim_size, dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggr}")

    def update(
        self,
        aggr_out: jnp.ndarray,
        x: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Update node embeddings after aggregation.

        Args:
            aggr_out: Aggregated messages [num_nodes, features]
            x: Original node features [num_nodes, features]

        Returns:
            Updated node features [num_nodes, features]
        """
        # Default: just return aggregated output
        return aggr_out

    def message_and_aggregate(
        self,
        x: Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        dim_size: int | None = None,
    ) -> jnp.ndarray:
        """Fused message and aggregation for efficiency.

        The base class provides no fused path and always raises; override this
        hook when message computation and aggregation can be expressed in a
        single pass, *e.g.* when a sum aggregation of linearly transformed
        neighbours does not need all messages materialized.
        :meth:`propagate` dispatches here only for subclasses that override it,
        and then calls neither :meth:`message` nor :meth:`aggregate`; it still
        passes the returned array through :meth:`update`.

        An override receives the arguments of :meth:`propagate` untouched: ``x``
        is the whole node feature table -- or the ``(x_src, x_dst)`` tuple, with
        ``x[0]`` the source set -- and not the per-edge gather, and
        ``edge_index`` is the raw edge index, whose rows are ``(source, target)``
        for :obj:`flow="source_to_target"` and ``(target, source)`` otherwise.
        Gathering the endpoints and honouring :attr:`flow` is therefore the
        override's own job.

        Args:
            x: Node features, or a ``(x_src, x_dst)`` tuple for bipartite graphs
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_features]
            dim_size: Number of target nodes to scatter into

        Returns:
            Aggregated messages [dim_size, features]

        Raises:
            NotImplementedError: Always, as the base class provides no fused path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a fused `message_and_aggregate`"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the message passing layer.

        This default forwards its arguments straight to :meth:`propagate`, whose
        signature is ``(edge_index, x, edge_attr=None, size=None)``.

        Every concrete layer overrides this with its own signature, and those
        signatures genuinely differ: :class:`GCNConv` takes ``edge_weight``,
        :class:`GATConv` and :class:`GATv2Conv` return a ``(features, attention)``
        tuple when asked for attention weights, and :class:`EdgeConv` accepts
        precomputed neighbour indices. The layers are therefore not substitutable
        for one another, and this base signature deliberately imposes no contract
        beyond "callable". Consult the concrete layer's own annotations.

        Args:
            *args: Forwarded to :meth:`propagate`.
            **kwargs: Forwarded to :meth:`propagate`.

        Returns:
            Updated node features.
        """
        return self.propagate(*args, **kwargs)


def create_edge_index_with_padding(
    edge_index: jnp.ndarray,
    num_nodes: int,
    max_edges: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create padded edge indices for fixed-size batching.

    This is useful for JAX operations that require fixed shapes
    for JIT compilation.

    Args:
        edge_index: Original edge indices [2, num_edges]
        num_nodes: Number of nodes in the graph
        max_edges: Maximum number of edges (for padding)

    Returns:
        Tuple of (padded_edge_index, edge_mask)
    """
    num_edges = edge_index.shape[1]

    if num_edges >= max_edges:
        # Truncate if necessary
        return edge_index[:, :max_edges], jnp.ones(max_edges, dtype=jnp.bool_)

    # Pad with self-loops on node 0 (these will be masked out)
    padding_needed = max_edges - num_edges
    padding = jnp.zeros((2, padding_needed), dtype=edge_index.dtype)

    padded_edge_index = jnp.concatenate([edge_index, padding], axis=1)
    edge_mask = jnp.concatenate(
        [
            jnp.ones(num_edges, dtype=jnp.bool_),
            jnp.zeros(padding_needed, dtype=jnp.bool_),
        ]
    )

    return padded_edge_index, edge_mask
