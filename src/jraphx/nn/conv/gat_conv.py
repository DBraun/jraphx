"""Graph Attention Network (GAT) layer implementation."""

from typing import Union

from flax.nnx import Dropout, Linear, Param, Rngs, initializers
from jax import numpy as jnp

from jraphx.nn.conv.message_passing import MessagePassing
from jraphx.utils import scatter_add, scatter_softmax
from jraphx.utils.loop import add_self_loops as add_self_loops_fn


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j}\mathbf{\Theta}_t\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t}\mathbf{\Theta}_{t}\mathbf{x}_k
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_j
        + \mathbf{a}^{\top}_{e} \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_k
        + \mathbf{a}^{\top}_{e} \mathbf{\Theta}_{e} \mathbf{e}_{i,k}
        \right)\right)}.

    If the graph is not bipartite, :math:`\mathbf{\Theta}_{s} =
    \mathbf{\Theta}_{t}`.

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
        fill_value (float or str, optional): The way to generate edge features of
            self-loops (in case :obj:`edge_dim != None`).
            (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
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
        fill_value: Union[float, str] = "mean",
        bias: bool = True,
        residual: bool = False,
        rngs: Rngs | None = None,
    ):
        """Initialize the GAT layer."""
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
        self.residual = residual

        # Handle bipartite graphs
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_features, int):
            self.lin = Linear(
                in_features,
                heads * out_features,
                use_bias=False,
                rngs=rngs,
            )
        else:
            # Bipartite graph with different source and target features
            self.lin_src = Linear(
                in_features[0],
                heads * out_features,
                use_bias=False,
                rngs=rngs,
            )
            self.lin_dst = Linear(
                in_features[1],
                heads * out_features,
                use_bias=False,
                rngs=rngs,
            )

        # Attention parameters for each head
        self.att_src = Param(initializers.glorot_uniform()(rngs.params(), (heads, out_features)))
        self.att_dst = Param(initializers.glorot_uniform()(rngs.params(), (heads, out_features)))

        # Edge feature transformation and attention
        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim,
                heads * out_features,
                use_bias=False,
                rngs=rngs,
            )
            self.att_edge = Param(
                initializers.glorot_uniform()(rngs.params(), (heads, out_features))
            )
        else:
            self.lin_edge = None
            self.att_edge = None

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

        # Bias
        if bias:
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
        size: tuple[int, int] | None = None,
        return_attention_weights: bool = False,
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]]:
        """Forward pass of the GAT layer.

        Args:
            x: Node features [num_nodes, in_features] or tuple for bipartite graphs
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            size: Optional size (num_src_nodes, num_dst_nodes) for bipartite graphs
            return_attention_weights: If True, return attention weights

        Returns:
            Updated node features [num_nodes, heads * out_features] if concat
            or [num_nodes, out_features] if not concat.
            If return_attention_weights is True, also returns (out, (edge_index, alpha)).
        """
        # Handle bipartite graphs
        res = None
        if isinstance(x, tuple):
            x_src, x_dst = x
            # Handle case where x_dst is None (source nodes only)
            num_nodes = x_dst.shape[0] if x_dst is not None else x_src.shape[0]
            # Override with size parameter if provided
            if size is not None:
                num_nodes = size[1]  # Target node count

            # Residual connection for target nodes
            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            # Linear transformation
            if self.lin is not None:
                # Non-bipartite initialization used on bipartite data
                x_src = self.lin(x_src)
                x_dst = self.lin(x_dst) if x_dst is not None else None
            else:
                x_src = self.lin_src(x_src) if self.lin_src is not None else x_src
                x_dst = (
                    self.lin_dst(x_dst) if self.lin_dst is not None and x_dst is not None else None
                )

            x_src = x_src.reshape(-1, self.heads, self.out_features)
            if x_dst is not None:
                x_dst = x_dst.reshape(-1, self.heads, self.out_features)
        else:
            num_nodes = x.shape[0]
            # Override with size parameter if provided
            if size is not None:
                num_nodes = size[1]  # Target node count

            # Residual connection
            if self.res is not None:
                res = self.res(x)

            # Linear transformation and reshape to separate heads
            if self.lin is not None:
                x = self.lin(x)
            else:
                # Bipartite initialization used on non-bipartite data
                x = self.lin_dst(x) if self.lin_dst is not None else x
            x = x.reshape(num_nodes, self.heads, self.out_features)
            x_src = x_dst = x

        # Add self-loops
        if self._add_self_loops:
            edge_index, edge_attr = add_self_loops_fn(
                edge_index, edge_attr=edge_attr, fill_value=self.fill_value, num_nodes=num_nodes
            )

        # Compute attention scores
        row, col = edge_index[0], edge_index[1]

        # Get source and target features for edges
        x_i = x_dst[col] if x_dst is not None else x_src[col]  # [num_edges, heads, out_features]
        x_j = x_src[row]  # [num_edges, heads, out_features]

        # Compute attention scores for each head
        # e_{ij} = a_src^T x_i + a_dst^T x_j + (optional) a_edge^T edge_features
        alpha_src = jnp.sum(x_i * self.att_src.value, axis=-1)  # [num_edges, heads]
        alpha_dst = jnp.sum(x_j * self.att_dst.value, axis=-1)  # [num_edges, heads]
        alpha = alpha_src + alpha_dst  # [num_edges, heads]

        # Add edge feature attention if available
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.reshape(-1, 1)
            edge_feat = self.lin_edge(edge_attr)
            edge_feat = edge_feat.reshape(-1, self.heads, self.out_features)
            alpha_edge = jnp.sum(edge_feat * self.att_edge.value, axis=-1)
            alpha = alpha + alpha_edge

        # Apply LeakyReLU
        alpha = jnp.where(alpha > 0, alpha, alpha * self.negative_slope)

        # Compute softmax over neighbors for each node using our optimized scatter_softmax
        # This handles different numbers of neighbors efficiently
        # We need to flatten heads dimension for scatter_softmax
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

        # Aggregate messages using optimized scatter
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
