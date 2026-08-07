from typing import Literal

import jax.numpy as jnp
from flax import nnx

from jraphx.nn.conv.message_passing import MessagePassing
from jraphx.utils import scatter_softmax


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_features (int): Size of each input sample. Bipartite
            :obj:`(x_src, x_dst)` inputs are not supported; every node uses the
            same feature table for queries, keys and values.
        out_features (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [\mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1 \mathbf{x}_i - \mathbf{m}_i])`.
            Requires :obj:`root_weight=True`; without the skip term there is
            nothing to gate against, so :obj:`beta` is ignored.
            (default: :obj:`False`)
        dropout_rate (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left( \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij} \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        rngs: Random number generators for initialization.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})`
          where :math:`H` is the number of heads.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout_rate: float = 0.0,
        edge_dim: int | None = None,
        beta: bool = False,
        root_weight: bool = True,
        aggr: Literal["add", "mean", "max", "min"] = "add",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(aggr=aggr)

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.edge_dim = edge_dim
        # Beta gates the aggregation against the skip term, so it is meaningless
        # without one; with root_weight=False the flag is ignored.
        self.beta = beta and root_weight
        self.root_weight = root_weight

        # Single linear transformation for queries, keys, and values.
        # This is more efficient than three separate layers; the fused bias
        # splits into the per-projection biases along the same axis.
        self.lin_qkv = nnx.Linear(in_features, 3 * heads * out_features, use_bias=True, rngs=rngs)

        # Edge feature projection
        self.lin_edge: nnx.Linear | None
        if edge_dim is not None:
            self.lin_edge = nnx.Linear(edge_dim, heads * out_features, use_bias=False, rngs=rngs)
        else:
            self.lin_edge = nnx.data(None)

        # The skip and beta maps act on the output after heads are concatenated
        # or averaged, so their width follows the concat mode.
        skip_features = heads * out_features if concat else out_features

        # Skip connection transformation
        self.lin_skip: nnx.Linear | None
        if root_weight:
            self.lin_skip = nnx.Linear(in_features, skip_features, use_bias=True, rngs=rngs)
        else:
            self.lin_skip = nnx.data(None)

        # Beta gating parameter
        self.lin_beta: nnx.Linear | None
        if self.beta:
            self.lin_beta = nnx.Linear(3 * skip_features, 1, use_bias=False, rngs=rngs)
        else:
            self.lin_beta = nnx.data(None)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass of TransformerConv.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim], required if ``edge_dim`` was set

        Returns:
            Updated node features [num_nodes, out_features * heads] if concat
            else [num_nodes, out_features]

        Raises:
            RuntimeError: If the layer was built with ``edge_dim`` but ``edge_attr`` is None.
        """
        H, C = self.heads, self.out_features

        # Compute queries, keys, values with a single linear transformation
        qkv = self.lin_qkv(x)  # [N, 3*H*C]
        # Split into Q, K, V
        query, key, value = jnp.split(qkv, 3, axis=-1)  # Each is [N, H*C]

        # Propagate with attention - need custom propagation
        out = self._propagate_transformer(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr
        )

        # Average or concatenate heads before the skip term: the skip and beta
        # maps are sized for the final output width
        if not self.concat:
            out = out.reshape(-1, H, C)
            out = jnp.mean(out, axis=1)

        if self.lin_skip is not None:
            root = self.lin_skip(x)
            if self.lin_beta is not None:
                beta_input = jnp.concatenate([root, out, root - out], axis=-1)
                beta = nnx.sigmoid(self.lin_beta(beta_input))
                out = beta * root + (1 - beta) * out
            else:
                out = out + root

        return out

    def _propagate_transformer(
        self,
        edge_index: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Custom propagation for transformer attention."""
        # Get source and target indices
        row, col = edge_index[0], edge_index[1]

        # Gather features for edges
        query_i = jnp.take(query, col, axis=0)  # Target queries
        key_j = jnp.take(key, row, axis=0)  # Source keys
        value_j = jnp.take(value, row, axis=0)  # Source values

        # Compute messages with attention
        messages = self._attention_message(
            query_i=query_i,
            key_j=key_j,
            value_j=value_j,
            edge_attr=edge_attr,
            index=col,
            ptr=None,
            size_i=query.shape[0],
        )

        # Aggregate messages
        out = self.aggregate(messages, col, query.shape[0])

        return out

    def _attention_message(
        self,
        query_i: jnp.ndarray,
        key_j: jnp.ndarray,
        value_j: jnp.ndarray,
        index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        ptr: jnp.ndarray | None = None,
        size_i: int | None = None,
    ) -> jnp.ndarray:
        """Compute messages with attention weights.

        This layer computes attention over pre-projected query/key/value tensors
        rather than raw node features, so it does not implement
        :meth:`MessagePassing.message` and is not routed through
        :meth:`MessagePassing.propagate`.

        Args:
            query_i: Query features of target nodes [E, H*C]
            key_j: Key features of source nodes [E, H*C]
            value_j: Value features of source nodes [E, H*C]
            edge_attr: Edge features [E, edge_dim], required if ``edge_dim`` was set
            index: Target node indices for edges [E]
            ptr: Batch pointers (unused)
            size_i: Number of target nodes

        Returns:
            Weighted messages [E, H*C]

        Raises:
            RuntimeError: If the layer was built with ``edge_dim`` but ``edge_attr`` is None.
        """
        H, C = self.heads, self.out_features

        # Reshape to separate heads
        query_i = query_i.reshape(-1, H, C)
        key_j = key_j.reshape(-1, H, C)
        value_j = value_j.reshape(-1, H, C)

        # Add the projected edge features to both keys and values, so that edge
        # information conditions the attention scores as well as the messages
        if self.lin_edge is not None:
            if edge_attr is None:
                raise RuntimeError("edge_dim was set at construction, so edge_attr is required.")
            edge_feat = self.lin_edge(edge_attr).reshape(-1, H, C)
            key_j = key_j + edge_feat
            value_j = value_j + edge_feat

        # Compute attention scores
        alpha = (query_i * key_j).sum(axis=-1) / jnp.sqrt(C)  # [E, H]

        # Apply softmax to get attention weights
        alpha = scatter_softmax(alpha, index, dim=0, dim_size=size_i)

        # Apply dropout
        alpha = self.dropout(alpha)

        # Weight values by attention
        out = value_j * alpha.reshape(-1, H, 1)

        # Reshape back
        out = out.reshape(-1, H * C)

        return out
