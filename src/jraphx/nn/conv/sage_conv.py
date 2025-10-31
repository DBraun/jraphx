"""GraphSAGE convolution layer implementation."""

from typing import Literal, Union

import jax.numpy as jnp
from flax import nnx
from flax.nnx import Linear, Param, Rngs

from jraphx.nn.conv.message_passing import MessagePassing


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_features (int or tuple): Size of each input sample, or tuple for
            bipartite graphs. A tuple corresponds to the sizes of source and
            target dimensionalities.
        out_features (int): Size of each output sample.
        aggr (str, optional): The aggregation scheme to use.
            Can be :obj:`"mean"`, :obj:`"max"`, :obj:`"lstm"`, or :obj:`"gcn"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        rngs: Random number generators for initialization.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """

    def __init__(
        self,
        in_features: Union[int, tuple[int, int]],
        out_features: int,
        aggr: Literal["mean", "max", "lstm", "gcn"] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
        rngs: Rngs | None = None,
    ):
        """Initialize the GraphSAGE layer."""
        if aggr == "lstm":
            raise NotImplementedError("LSTM aggregation is not yet implemented")

        if aggr == "gcn":
            # GCN-style aggregation doesn't separate self and neighbor features
            super().__init__(aggr="add")
            root_weight = False
        else:
            super().__init__(aggr=aggr)

        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.root_weight = root_weight
        self.aggr_type = aggr

        # Handle bipartite graphs
        if isinstance(in_features, tuple):
            in_features_src = in_features[0]
            in_features_dst = in_features[1]
        else:
            in_features_src = in_features_dst = in_features

        # Linear transformation for neighbor features
        if aggr == "gcn":
            # GCN-style: single transformation for all features
            self.lin = Linear(
                in_features_src,
                out_features,
                use_bias=bias,
                rngs=rngs,
            )
            self.lin_r = nnx.data(None)
        else:
            # Separate transformations for neighbors and self
            self.lin = Linear(
                in_features_src,
                out_features,
                use_bias=False,  # Bias added at the end
                rngs=rngs,
            )

            # Linear transformation for root (self) features
            if root_weight:
                self.lin_r = Linear(
                    in_features_dst,
                    out_features,
                    use_bias=bias,
                    rngs=rngs,
                )
            else:
                self.lin_r = nnx.data(None)
                # Add bias manually if no root weight
                if bias:
                    self.bias = Param(jnp.zeros((out_features,)))
                else:
                    self.bias = nnx.data(None)

    def __call__(
        self,
        x: Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        """Forward pass of the GraphSAGE layer.

        Args:
            x: Node features or tuple of (source, target) features for bipartite graphs
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features (not used in GraphSAGE)
            size: Optional size (num_src_nodes, num_dst_nodes) for bipartite graphs

        Returns:
            Updated node features [num_nodes, out_features]
        """
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        if self.aggr_type == "gcn":
            # GCN-style aggregation
            # Transform features
            x_src = self.lin(x_src)

            # Propagate (includes self-loops by default in GCN)
            out = self.propagate(edge_index, x_src, edge_attr, size)
        else:
            # Standard GraphSAGE aggregation
            # Transform neighbor features
            x_j = self.lin(x_src)

            # Aggregate neighbor features
            out = self.propagate(edge_index, x_j, edge_attr, size)

            # Add transformed root features
            if self.root_weight and x_dst is not None:
                out = out + self.lin_r(x_dst)
            elif hasattr(self, "bias") and self.bias is not None:
                out = out + self.bias.value

        # L2 normalization
        if self.normalize:
            out = out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-10)

        return out

    def message(
        self,
        x_j: jnp.ndarray,
        x_i: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Construct messages from source nodes.

        Args:
            x_j: Source node features [num_edges, out_features]
            x_i: Target node features (not used)
            edge_attr: Edge features (not used)

        Returns:
            Messages [num_edges, out_features]
        """
        return x_j

    def aggregate(
        self,
        messages: jnp.ndarray,
        index: jnp.ndarray,
        dim_size: int | None = None,
    ) -> jnp.ndarray:
        """Aggregate messages based on the specified method.

        Args:
            messages: Messages to aggregate [num_edges, out_features]
            index: Target node indices [num_edges]
            dim_size: Number of target nodes

        Returns:
            Aggregated messages [num_nodes, out_features]
        """
        if self.aggr_type == "gcn":
            # GCN uses add aggregation with normalization
            # This should be handled by the parent class
            return super().aggregate(messages, index, dim_size)
        else:
            # Use parent class aggregation (mean or max)
            return super().aggregate(messages, index, dim_size)
