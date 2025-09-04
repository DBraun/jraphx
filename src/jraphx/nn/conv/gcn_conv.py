"""Graph Convolutional Network (GCN) layer implementation with optimizations."""

from flax.nnx import Linear, Rngs, Variable
from jax import numpy as jnp
from jax.ops import segment_sum

from jraphx.nn.conv.message_passing import MessagePassing
from jraphx.utils.degree import degree
from jraphx.utils.loop import add_self_loops as add_self_loops_fn
from jraphx.utils.num_nodes import maybe_num_nodes


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. By default, self-loops will be added
            when :obj:`normalize` is set to :obj:`True`.
            (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        rngs: Random number generators for initialization.
        static_num_nodes (int, optional): Optional static number of nodes for
            better JIT performance.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        rngs: Rngs | None = None,
        static_num_nodes: int | None = None,
    ):
        """Initialize the GCN layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            improved: If True, use improved GCN normalization
            cached: If True, cache normalized edge weights (disable for JIT)
            add_self_loops: If True, add self-loops to the graph
            normalize: If True, apply symmetric normalization
            bias: If True, add a learnable bias
            rngs: Random number generators for initialization
            static_num_nodes: Optional static number of nodes for better JIT performance
        """
        super().__init__(aggr="add")

        if add_self_loops and not normalize:
            raise ValueError(
                f"'{self.__class__.__name__}' does not support "
                f"adding self-loops to the graph when no "
                f"on-the-fly normalization is applied"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.improved = improved
        self.cached = cached
        self._add_self_loops = add_self_loops
        self.normalize = normalize
        self.static_num_nodes = static_num_nodes

        # Linear transformation
        self.linear = Linear(
            in_features,
            out_features,
            use_bias=bias,
            rngs=rngs,
        )

        # Cache for normalized edge weights (for static graphs)
        if cached:
            self._cached_edge_index = Variable(None)
            self._cached_edge_weight = Variable(None)
            self._cached_num_nodes = Variable(None)
        else:
            self._cached_edge_index = None
            self._cached_edge_weight = None
            self._cached_num_nodes = None

    def gcn_norm(
        self,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
        num_nodes: int | None = None,
        improved: bool = False,
        add_self_loops: bool = True,
        dtype: jnp.dtype | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply GCN normalization to edge weights with optimizations.

        This method uses efficient degree computation and caching when possible.

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            num_nodes: Number of nodes
            improved: Use improved normalization
            add_self_loops: Add self-loops
            dtype: Data type for edge weights

        Returns:
            Tuple of (edge_index, normalized edge_weight)
        """
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        dtype = dtype or jnp.float32

        if edge_weight is None:
            edge_weight = jnp.ones(edge_index.shape[1], dtype=dtype)

        # Add self-loops
        if add_self_loops:
            fill_value = 2.0 if improved else 1.0
            edge_index, edge_weight = add_self_loops_fn(
                edge_index,
                edge_weight,
                fill_value=fill_value,
                num_nodes=num_nodes,
            )

        # Compute normalization using optimized degree computation
        row, col = edge_index[0], edge_index[1]

        # Efficient degree computation
        deg = degree(col, num_nodes, dtype=dtype)

        # Compute inverse square root of degree
        # Use jnp.where for numerical stability
        deg_inv_sqrt = jnp.where(deg > 0, jnp.power(deg, -0.5), 0.0)

        # Apply normalization: D^{-1/2} A D^{-1/2}
        # Use JAX's take for efficient indexing
        norm_row = jnp.take(deg_inv_sqrt, row)
        norm_col = jnp.take(deg_inv_sqrt, col)
        edge_weight = norm_row * edge_weight * norm_col

        return edge_index, edge_weight

    def _get_cached_edge_weight(
        self,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None,
        num_nodes: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get cached edge weights or compute and cache them.

        Args:
            edge_index: Edge indices
            edge_weight: Optional edge weights
            num_nodes: Number of nodes

        Returns:
            Tuple of (edge_index, edge_weight)
        """
        if not self.cached:
            # No caching, compute fresh
            return self.gcn_norm(
                edge_index,
                edge_weight,
                num_nodes,
                self.improved,
                self._add_self_loops,
            )

        # Check if cache is valid
        cache_valid = (
            self._cached_edge_index.value is not None
            and self._cached_num_nodes.value == num_nodes
            and jnp.array_equal(self._cached_edge_index.value, edge_index)
        )

        if cache_valid:
            # Return cached values
            return self._cached_edge_index.value, self._cached_edge_weight.value

        # Compute and cache
        edge_index, edge_weight = self.gcn_norm(
            edge_index,
            edge_weight,
            num_nodes,
            self.improved,
            self._add_self_loops,
        )

        # Update cache
        self._cached_edge_index.value = edge_index
        self._cached_edge_weight.value = edge_weight
        self._cached_num_nodes.value = num_nodes

        return edge_index, edge_weight

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass of the GCN layer with optimizations.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' received a tuple "
                f"of node features as input while this layer "
                f"does not support bipartite message passing. "
                f"Please try other layers such as 'SAGEConv' instead"
            )

        # Apply linear transformation first (more cache-friendly)
        x = self.linear(x)

        # Use static_num_nodes if provided for better JIT performance
        num_nodes = self.static_num_nodes if self.static_num_nodes is not None else x.shape[0]

        # Get normalized edge weights (with caching if enabled)
        if self.normalize:
            if self.cached:
                edge_index, edge_weight = self._get_cached_edge_weight(
                    edge_index, edge_weight, num_nodes
                )
            else:
                edge_index, edge_weight = self.gcn_norm(
                    edge_index,
                    edge_weight,
                    num_nodes,
                    self.improved,
                    self._add_self_loops,
                    x.dtype,
                )
        elif self._add_self_loops:
            fill_value = 2.0 if self.improved else 1.0
            edge_index, edge_weight = add_self_loops_fn(
                edge_index,
                edge_weight,
                fill_value=fill_value,
                num_nodes=num_nodes,
            )

        # Message passing with edge weights
        if edge_weight is not None:
            # Efficient weighted aggregation
            row, col = edge_index[0], edge_index[1]
            # Use take for efficient indexing
            messages = jnp.take(x, row, axis=0) * edge_weight.reshape(-1, 1)
            # Use segment_sum for efficient aggregation
            out = segment_sum(
                messages,
                col,
                num_segments=num_nodes,
            )
        else:
            # Unweighted aggregation
            out = self.propagate(edge_index, x)

        return out

    def reset_cache(self):
        """Reset the cached edge weights.

        Call this when the graph structure changes.
        """
        if self.cached:
            self._cached_edge_index.value = None
            self._cached_edge_weight.value = None
            self._cached_num_nodes.value = None
