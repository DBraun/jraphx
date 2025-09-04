"""Base message passing class for graph neural networks in JAX/NNX.

This module provides an optimized base class for message passing operations
using JAX's efficient indexing and gathering operations.
"""

from typing import Literal, Union

import jax.numpy as jnp
from flax.nnx import Module

from jraphx.utils.scatter import scatter_add, scatter_max, scatter_mean, scatter_min


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

        This method uses optimized JAX operations for efficient indexing
        and gathering of node features.

        Args:
            edge_index: Edge indices [2, num_edges]
            x: Node features [num_nodes, features] or tuple for bipartite graphs
            edge_attr: Optional edge features [num_edges, edge_features]
            size: Optional size (num_src_nodes, num_dst_nodes) for bipartite graphs

        Returns:
            Updated node features after message passing
        """
        # Handle bipartite vs regular graphs
        if isinstance(x, tuple):
            x_i, x_j = x
            size = size or (x_j.shape[0], x_i.shape[0])
        else:
            x_i = x_j = x
            # If size is explicitly provided, use it (for bipartite cases)
            if size is None:
                size = (x.shape[0], x.shape[0])

        # Get source and target indices based on flow
        if self.flow == "source_to_target":
            row, col = edge_index[0], edge_index[1]
        else:
            row, col = edge_index[1], edge_index[0]

        # Use efficient JAX indexing for gathering node features
        # jnp.take is more efficient than direct indexing for large arrays
        x_j_gathered = jnp.take(x_j, row, axis=0)  # Source nodes
        x_i_gathered = jnp.take(x_i, col, axis=0)  # Target nodes

        # Compute messages
        messages = self.message(x_j_gathered, x_i_gathered, edge_attr)

        # Aggregate messages
        dim_size = size[1] if self.flow == "source_to_target" else size[0]

        # Check if we can use fused message_and_aggregate
        if hasattr(self, "message_and_aggregate") and self.aggr in [
            "add",
            "mean",
            "max",
            "min",
        ]:
            # Use fused operation if available
            aggr_out = self.message_and_aggregate(x_j, edge_index, edge_attr, dim_size)
        else:
            aggr_out = self.aggregate(messages, col, dim_size)

        # Update node embeddings
        if isinstance(x, tuple):
            x_original = x[1] if self.flow == "source_to_target" else x[0]
        else:
            x_original = x

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
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        dim_size: int | None = None,
    ) -> jnp.ndarray:
        """Fused message and aggregation for efficiency.

        This can be overridden for more efficient implementations
        when message computation and aggregation can be fused.
        For example, for simple aggregations like sum/mean with
        linear transformations, we can avoid materializing all messages.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge features

        Returns:
            Aggregated messages
        """
        # Get indices based on flow
        if self.flow == "source_to_target":
            row, col = edge_index[0], edge_index[1]
        else:
            row, col = edge_index[1], edge_index[0]

        # Efficient gathering using JAX operations
        x_j = jnp.take(x, row, axis=0)
        x_i = jnp.take(x, col, axis=0)

        messages = self.message(x_j, x_i, edge_attr)
        return self.aggregate(messages, col, dim_size or x.shape[0])

    def __call__(
        self,
        x: Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        """Forward pass through the message passing layer.

        Args:
            x: Node features or tuple for bipartite graphs
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features
            size: Optional size for bipartite graphs

        Returns:
            Updated node features
        """
        return self.propagate(edge_index, x, edge_attr, size)


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
