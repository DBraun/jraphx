"""Graph Convolutional Network (GCN) layer implementation with optimizations."""

from flax import nnx
from flax.nnx import Linear, Param, Rngs, Variable
from jax import numpy as jnp
from jax.ops import segment_sum

from jraphx.nn.conv.message_passing import MessagePassing
from jraphx.utils.loop import add_self_loops as add_self_loops_fn
from jraphx.utils.num_nodes import maybe_num_nodes


def _add_remaining_self_loops(
    edge_index: jnp.ndarray,
    edge_weight: jnp.ndarray,
    fill_value: float,
    num_nodes: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Give every node exactly one self-loop, keeping the weight it already has.

    One loop :math:`(i, i)` is appended for each of the ``num_nodes`` nodes. An
    edge of the input that already is a self-loop keeps its weight: that weight
    is carried over to the appended loop and the original entry is set to zero,
    so a node's loop is counted exactly once in the degree. Nodes without a
    self-loop get ``fill_value``. The result always has
    ``num_edges + num_nodes`` edges, which keeps the transformation traceable by
    :obj:`jax.jit`.

    Args:
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]
        fill_value: Weight given to the self-loop of a node that has none
        num_nodes: Number of nodes

    Returns:
        Tuple of (edge indices with self-loops, edge weights with self-loops)
    """
    row = edge_index[0]
    is_loop = row == edge_index[1]

    # Scatter positions outside of the array are dropped, so only pre-existing
    # self-loops overwrite the fill value of their node.
    loop_weight = jnp.full(num_nodes, fill_value, dtype=edge_weight.dtype)
    loop_weight = loop_weight.at[jnp.where(is_loop, row, num_nodes)].set(edge_weight, mode="drop")

    edge_weight = jnp.where(is_loop, jnp.zeros_like(edge_weight), edge_weight)

    loop_nodes = jnp.arange(num_nodes, dtype=edge_index.dtype)
    edge_index = jnp.concatenate([edge_index, jnp.stack([loop_nodes, loop_nodes])], axis=1)
    edge_weight = jnp.concatenate([edge_weight, loop_weight])

    return edge_index, edge_weight


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
        cached (bool, optional): If set to :obj:`True`, the layer reuses a
            precomputed :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` instead of normalizing on every call.
            The cache is filled by :meth:`precompute_norm`, which must be
            called once, eagerly (outside of :obj:`jax.jit`/:obj:`nnx.jit`),
            before the first forward pass; a forward pass with an empty cache
            raises :obj:`RuntimeError`. Mutating module state from inside a
            JAX transformation is not possible, so the cache is never filled
            implicitly. Caching only applies when :obj:`normalize` is
            :obj:`True`, and should only be used in transductive learning
            scenarios where the graph never changes.
            (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. By default, self-loops will be added
            when :obj:`normalize` is set to :obj:`True`. A node that already
            carries a self-loop keeps that loop's weight instead of receiving a
            second one. (default: :obj:`True`)
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
        static_num_nodes: int | None = None,
        *,
        rngs: Rngs,
    ):
        """Initialize the GCN layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            improved: If True, use improved GCN normalization
            cached: If True, reuse the normalization filled by precompute_norm()
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
            use_bias=False,  # Bias added after aggregation
            rngs=rngs,
        )

        self.bias: Param | None
        if bias:
            self.bias = Param(jnp.zeros((out_features,)))
        else:
            self.bias = nnx.data(None)

        # Cache for normalized edge weights (for static graphs). The variables hold
        # None until `precompute_norm` fills them.
        self._cached_edge_index: Variable[jnp.ndarray | None] | None
        self._cached_edge_weight: Variable[jnp.ndarray | None] | None
        if cached:
            self._cached_edge_index = Variable(None)
            self._cached_edge_weight = Variable(None)
        else:
            self._cached_edge_index = nnx.data(None)
            self._cached_edge_weight = nnx.data(None)

        # Kept as a plain Python attribute so that it stays static under
        # tracing and can guard the cache against a change of graph size.
        self._cached_num_nodes: int | None = None

    def gcn_norm(
        self,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
        num_nodes: int | None = None,
        improved: bool = False,
        add_self_loops: bool = True,
        dtype: jnp.dtype | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply symmetric GCN normalization to edge weights.

        The normalized weight of edge :math:`(j, i)` is
        :math:`e_{j,i} / \\sqrt{\\hat{d}_j \\hat{d}_i}`, where
        :math:`\\hat{d}_i` is the *weighted* in-degree of node :math:`i`
        computed over the edge weights after self-loop insertion. Nodes with a
        non-positive weighted degree receive a normalization factor of zero.

        Every node ends up with exactly one self-loop. A self-loop that is
        already present in :attr:`edge_index` keeps its own weight and is not
        duplicated: its entry is zeroed out and its weight moves to the appended
        loop, so it enters the degree once instead of twice.

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

        # Add self-loops, without duplicating the ones already present
        if add_self_loops:
            fill_value = 2.0 if improved else 1.0
            edge_index, edge_weight = _add_remaining_self_loops(
                edge_index,
                edge_weight,
                fill_value,
                num_nodes,
            )

        # Compute normalization using optimized degree computation
        row, col = edge_index[0], edge_index[1]

        # Weighted in-degree over the (self-looped) edge weights
        deg = segment_sum(edge_weight, col, num_segments=num_nodes).astype(dtype)

        # Compute inverse square root of degree
        # Use jnp.where for numerical stability
        deg_inv_sqrt = jnp.where(deg > 0, jnp.power(deg, -0.5), 0.0)

        # Apply normalization: D^{-1/2} A D^{-1/2}
        # Use JAX's take for efficient indexing
        norm_row = jnp.take(deg_inv_sqrt, row)
        norm_col = jnp.take(deg_inv_sqrt, col)
        edge_weight = norm_row * edge_weight * norm_col

        return edge_index, edge_weight

    def precompute_norm(
        self,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
        num_nodes: int | None = None,
        dtype: jnp.dtype | None = None,
    ) -> None:
        """Fill the normalization cache for a fixed graph.

        Must be called eagerly, i.e. outside of any JAX transformation, since it
        mutates module state. Afterwards the layer can be called under
        :obj:`jax.jit` or :obj:`nnx.jit` and reuses the stored normalization
        instead of recomputing it. Call :meth:`reset_cache` before precomputing
        a different graph.

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            num_nodes: Number of nodes, defaults to :obj:`static_num_nodes` or
                the largest node index in ``edge_index`` plus one
            dtype: Data type of the normalized edge weights

        Raises:
            ValueError: If the layer was not constructed with :obj:`cached=True`.
        """
        if self._cached_edge_index is None or self._cached_edge_weight is None:
            raise ValueError(
                f"'{self.__class__.__name__}.precompute_norm()' requires "
                f"'cached=True'; a layer with 'cached=False' normalizes on every "
                f"forward pass"
            )

        if num_nodes is None:
            num_nodes = self.static_num_nodes
        num_nodes = int(maybe_num_nodes(edge_index, num_nodes))

        edge_index, edge_weight = self.gcn_norm(
            edge_index,
            edge_weight,
            num_nodes,
            self.improved,
            self._add_self_loops,
            dtype,
        )

        self._cached_edge_index.set_value(edge_index)
        self._cached_edge_weight.set_value(edge_weight)
        self._cached_num_nodes = num_nodes

    def _get_cached_edge_weight(self, num_nodes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Read the precomputed normalization.

        Args:
            num_nodes: Number of nodes of the graph being processed

        Returns:
            Tuple of (edge_index with self-loops, normalized edge_weight)

        Raises:
            RuntimeError: If the cache is empty or was built for a different
                number of nodes.
        """
        if self._cached_edge_index is None or self._cached_edge_weight is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' was constructed with 'cached=False', so "
                f"it holds no normalization cache"
            )

        cached_edge_index = self._cached_edge_index.get_value()
        cached_edge_weight = self._cached_edge_weight.get_value()
        if cached_edge_index is None or cached_edge_weight is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' was constructed with 'cached=True' but "
                f"its normalization cache is empty. Call "
                f"'precompute_norm(edge_index, edge_weight, num_nodes)' once, outside "
                f"of any JAX transformation, or use 'cached=False'"
            )

        if self._cached_num_nodes != num_nodes:
            raise RuntimeError(
                f"'{self.__class__.__name__}' cached a normalization for "
                f"{self._cached_num_nodes} nodes but received a graph with "
                f"{num_nodes} nodes. Call 'reset_cache()' and 'precompute_norm()' "
                f"for the new graph"
            )

        return cached_edge_index, cached_edge_weight

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
                edge_index, edge_weight = self._get_cached_edge_weight(num_nodes)
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

        # Bias is a constant offset, so it is added after aggregation
        if self.bias is not None:
            out = out + self.bias[...]

        return out

    def reset_cache(self) -> None:
        """Reset the cached edge weights.

        Call this when the graph structure changes.
        """
        if self._cached_edge_index is not None and self._cached_edge_weight is not None:
            self._cached_edge_index.set_value(None)
            self._cached_edge_weight.set_value(None)
        self._cached_num_nodes = None
