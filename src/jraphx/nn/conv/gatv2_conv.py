"""Graph Attention Network v2 (GATv2) layer implementation."""

from typing import Union

from flax.nnx import Dropout, Linear, Param, Rngs, initializers, leaky_relu
from jax import numpy as jnp

from jraphx.nn.conv.message_passing import MessagePassing
from jraphx.utils import scatter_add, scatter_softmax
from jraphx.utils.loop import add_self_loops as add_self_loops_fn


class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard GAT layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in GATv2, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,k}
        \right)\right)}.

    Args:
        in_features (int or tuple): Size of each input sample, or tuple for
            bipartite graphs. A tuple corresponds to the sizes of source and
            target dimensionalities.
        out_features (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float, optional): The way to generate edge features of
            self-loops (in case :obj:`edge_dim != None`).
            (default: :obj:`0.0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        residual (bool, optional): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)
        rngs: Random number generators for initialization.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})`
          where :math:`H` is the number of heads.
    """

    def __init__(
        self,
        in_features: Union[int, tuple[int, int]],
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: int | None = None,
        fill_value: float = 0.0,
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        rngs: Rngs | None = None,
    ):
        """Initialize the GATv2 layer."""
        super().__init__(aggr="add")

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout
        self._add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.residual = residual

        # Linear transformations
        if isinstance(in_features, int):
            self.lin_l = Linear(
                in_features,
                heads * out_features,
                use_bias=bias,
                rngs=rngs,
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_features,
                    heads * out_features,
                    use_bias=bias,
                    rngs=rngs,
                )
        else:
            # Bipartite graph with different source and target features
            self.lin_l = Linear(
                in_features[0],
                heads * out_features,
                use_bias=bias,
                rngs=rngs,
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_features[1],
                    heads * out_features,
                    use_bias=bias,
                    rngs=rngs,
                )

        # Attention parameter (single vector per head)
        self.att = Param(initializers.glorot_uniform()(rngs.params(), (heads, out_features)))

        # Edge feature transformation
        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim,
                heads * out_features,
                use_bias=False,
                rngs=rngs,
            )
        else:
            self.lin_edge = None

        # Residual connection
        total_out_features = heads * out_features if concat else out_features
        if residual:
            res_in_features = in_features if isinstance(in_features, int) else in_features[1]
            self.res = Linear(
                res_in_features,
                total_out_features,
                use_bias=False,
                rngs=rngs,
            )
        else:
            self.res = None

        # Bias (applied after aggregation)
        if bias and not isinstance(in_features, int):
            # For bipartite graphs, bias is handled by lin_l and lin_r
            self.bias = None
        elif bias:
            self.bias = Param(jnp.zeros((total_out_features,)))
        else:
            self.bias = None

        # Dropout
        if dropout > 0:
            self.dropout = Dropout(dropout, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        return_attention_weights: bool = False,
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]]:
        """Forward pass of the GATv2 layer.

        Args:
            x: Node features [num_nodes, in_features] or tuple for bipartite graphs
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            return_attention_weights: If True, return attention weights

        Returns:
            Updated node features [num_nodes, heads * out_features] if concat
            or [num_nodes, out_features] if not concat.
            If return_attention_weights is True, also returns (out, (edge_index, alpha)).
        """
        H, C = self.heads, self.out_features

        # Handle input types and compute residual
        res = None
        if isinstance(x, tuple):
            x_l, x_r = x
            num_nodes = x_r.shape[0] if x_r is not None else x_l.shape[0]

            # Residual connection for target nodes
            if self.res is not None and x_r is not None:
                res = self.res(x_r)

            # Linear transformation
            x_l = self.lin_l(x_l).reshape(-1, H, C)
            x_r = self.lin_r(x_r).reshape(-1, H, C) if x_r is not None else x_l
        else:
            num_nodes = x.shape[0]

            # Residual connection
            if self.res is not None:
                res = self.res(x)

            # Linear transformation
            x_l = self.lin_l(x).reshape(-1, H, C)
            x_r = self.lin_r(x).reshape(-1, H, C)

        # Add self-loops
        if self._add_self_loops:
            edge_index, edge_attr = add_self_loops_fn(
                edge_index, edge_attr=edge_attr, fill_value=self.fill_value, num_nodes=num_nodes
            )

        # Get edge endpoints
        row, col = edge_index[0], edge_index[1]

        # Get source and target features for edges
        x_i = x_r[col]  # [num_edges, heads, out_features]
        x_j = x_l[row]  # [num_edges, heads, out_features]

        # Key difference from GAT: combine features BEFORE applying attention
        x_combined = x_i + x_j  # [num_edges, heads, out_features]

        # Add edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.reshape(-1, 1)
            edge_feat = self.lin_edge(edge_attr)
            edge_feat = edge_feat.reshape(-1, H, C)
            x_combined = x_combined + edge_feat

        # Apply LeakyReLU (this is the key difference - applied after combination)
        x_combined = leaky_relu(x_combined, negative_slope=self.negative_slope)

        # Compute attention scores
        alpha = jnp.sum(x_combined * self.att.value, axis=-1)  # [num_edges, heads]

        # Apply softmax using our optimized scatter_softmax
        num_edges = alpha.shape[0]
        alpha_flat = alpha.reshape(-1)  # [num_edges * heads]

        # Create expanded index for each head
        col_expanded = jnp.repeat(col, self.heads)

        # Apply softmax
        alpha_flat = scatter_softmax(alpha_flat, col_expanded, dim_size=num_nodes)
        alpha = alpha_flat.reshape(num_edges, self.heads)

        # Apply dropout to attention coefficients
        if self.dropout is not None:
            alpha = self.dropout(alpha)

        # Apply attention weights to features
        weighted_features = x_j * alpha.reshape(
            -1, self.heads, 1
        )  # [num_edges, heads, out_features]

        # Aggregate messages
        weighted_features_flat = weighted_features.reshape(-1, self.heads * self.out_features)
        out_flat = scatter_add(weighted_features_flat, col, dim_size=num_nodes)
        out = out_flat.reshape(num_nodes, self.heads, self.out_features)

        # Concatenate or average heads
        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(axis=1)

        # Add residual connection
        if res is not None:
            out = out + res

        # Add bias
        if self.bias is not None:
            out = out + self.bias.value

        if return_attention_weights:
            return out, (edge_index, alpha)

        return out
