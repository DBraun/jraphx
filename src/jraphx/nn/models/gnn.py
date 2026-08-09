"""Pre-built GNN models (GCN, GAT, GraphSAGE, GIN)."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jraphx.nn.conv import GATConv, GATv2Conv, GCNConv, GINConv, MessagePassing, SAGEConv
from jraphx.nn.models.basic_gnn import BasicGNN
from jraphx.nn.models.mlp import MLP


class GCN(BasicGNN):
    """Graph Convolutional Network.

    From "Semi-supervised Classification with Graph Convolutional Networks"
    https://arxiv.org/abs/1609.02907

    Uses GCNConv layers for message passing.

    Args:
        in_features: Size of input features
        hidden_features: Size of hidden layers
        num_layers: Number of GCN layers
        out_features: Size of output (if None, uses hidden_features)
        dropout_rate: Dropout probability
        act: Non-linear activation function, or None to disable the activation
            entirely (default: jax.nn.relu)
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', 'graph_norm', None)
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        improved: Use improved GCN normalization
        cached: Reuse a precomputed normalization for a static graph. The cache
            of every layer must be filled eagerly with :meth:`precompute_norm`
            before the first forward pass; a forward pass with an empty cache
            raises RuntimeError.
        add_self_loops: Add self-loops to the graph
        normalize: Apply symmetric normalization
        rngs: Random number generators
    """

    supports_edge_weight: bool = True
    supports_edge_attr: bool = False

    def precompute_norm(
        self,
        edge_index: jax.Array,
        edge_weight: jax.Array | None = None,
        num_nodes: int | None = None,
        dtype: str | type | jnp.dtype | None = None,
    ) -> None:
        """Fill the normalization cache of every :class:`GCNConv` layer.

        Only meaningful for a model built with :obj:`cached=True`. Must be
        called eagerly, i.e. outside of any JAX transformation, since it mutates
        module state. Afterwards the model can be called under :obj:`jax.jit` or
        :obj:`nnx.jit` and reuses the stored normalization.

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            num_nodes: Number of nodes, defaults to the largest node index in
                ``edge_index`` plus one
            dtype: Data type of the normalized edge weights

        Raises:
            ValueError: If the model was not built with :obj:`cached=True`.
        """
        for conv in self.convs:
            if not isinstance(conv, GCNConv):
                raise RuntimeError(
                    f"Expected every layer of a GCN to be a GCNConv, found "
                    f"{type(conv).__name__}"
                )
            conv.precompute_norm(edge_index, edge_weight, num_nodes, dtype)

    def init_conv(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> MessagePassing:
        """Initialize GCNConv layer.

        Every keyword argument reaches the :class:`GCNConv` constructor, so an
        unsupported one raises a :obj:`TypeError` instead of being dropped.
        """
        return GCNConv(
            in_features,
            out_features,
            rngs=rngs,
            **kwargs,
        )


class GAT(BasicGNN):
    """Graph Attention Network.

    From "Graph Attention Networks" https://arxiv.org/abs/1710.10903
    or "How Attentive are Graph Attention Networks?" https://arxiv.org/abs/2105.14491

    Uses GATConv or GATv2Conv layers for message passing.

    Args:
        in_features: Size of input features
        hidden_features: Size of hidden layers (per head if concat=True)
        num_layers: Number of GAT layers
        out_features: Size of output (if None, uses hidden_features)
        heads: Number of attention heads
        concat: Whether to concatenate or average multi-head outputs
        v2: Use GATv2Conv instead of GATConv
        dropout_rate: Dropout probability
        act: Non-linear activation function, or None to disable the activation
            entirely (default: jax.nn.relu)
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', 'graph_norm', None)
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        edge_dim: Edge feature dimension
        rngs: Random number generators
    """

    supports_edge_weight: bool = False
    supports_edge_attr: bool = True

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        out_features: int | None = None,
        heads: int = 1,
        concat: bool = True,
        v2: bool = False,
        dropout_rate: float = 0.0,
        act: Callable | None = nnx.relu,
        act_first: bool = False,
        norm: str | None = None,
        jk: str | None = None,
        residual: bool = False,
        edge_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ):
        self.heads = heads
        self.concat = concat
        self.v2 = v2
        self.edge_dim = edge_dim

        # Adjust hidden_features for concatenation
        if concat:
            assert (
                hidden_features % heads == 0
            ), f"hidden_features ({hidden_features}) must be divisible by heads ({heads})"

        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            out_features=out_features,
            dropout_rate=dropout_rate,
            act=act,
            act_first=act_first,
            norm=norm,
            jk=jk,
            residual=residual,
            rngs=rngs,
            **kwargs,
        )

    def init_conv(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> MessagePassing:
        """Initialize GATConv or GATv2Conv layer.

        Remaining keyword arguments (:obj:`negative_slope`,
        :obj:`add_self_loops`, :obj:`fill_value`, ...) reach the convolution
        constructor, so an unsupported one raises a :obj:`TypeError` instead of
        being dropped.
        """
        Conv = GATv2Conv if self.v2 else GATConv

        # Concatenation is disabled only on a layer that maps straight to the
        # requested out_features; an embedding model (out_features=None) keeps
        # concatenating hidden_features // heads narrow heads on its last layer.
        is_last = len(self.convs) == self.num_layers - 1
        use_concat = self.concat and not (is_last and self._is_conv_to_out)

        if use_concat:
            # When concatenating, each head produces out_features/heads features
            head_features = out_features // self.heads
        else:
            # When averaging, each head produces out_features features
            head_features = out_features

        return Conv(
            in_features=in_features,
            out_features=head_features,
            heads=self.heads,
            concat=use_concat,
            # The model-level dropout rate also drops attention coefficients
            # inside each convolution, the GAT paper's primary regularizer.
            dropout=self.dropout_rate,
            edge_dim=self.edge_dim,
            residual=False,  # We handle residual in BasicGNN
            rngs=rngs,
            **kwargs,
        )


class GraphSAGE(BasicGNN):
    """GraphSAGE: Inductive Representation Learning on Large Graphs.

    From "Inductive Representation Learning on Large Graphs"
    https://arxiv.org/abs/1706.02216

    Uses SAGEConv layers for message passing.

    Args:
        in_features: Size of input features
        hidden_features: Size of hidden layers
        num_layers: Number of GraphSAGE layers
        out_features: Size of output (if None, uses hidden_features)
        aggr: Aggregation method ('mean', 'max', 'gcn')
        dropout_rate: Dropout probability
        act: Non-linear activation function, or None to disable the activation
            entirely (default: jax.nn.relu)
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', 'graph_norm', None)
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        normalize: Whether to L2-normalize output features
        rngs: Random number generators
    """

    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def init_conv(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> MessagePassing:
        """Initialize SAGEConv layer.

        Every keyword argument reaches the :class:`SAGEConv` constructor, so an
        unsupported one raises a :obj:`TypeError` instead of being dropped.
        """
        return SAGEConv(
            in_features,
            out_features,
            rngs=rngs,
            **kwargs,
        )


class GIN(BasicGNN):
    """Graph Isomorphism Network.

    From "How Powerful are Graph Neural Networks?"
    https://arxiv.org/abs/1810.00826

    Uses GINConv layers with MLP aggregation for message passing.

    Args:
        in_features: Size of input features
        hidden_features: Size of hidden layers
        num_layers: Number of GIN layers
        out_features: Size of output (if None, uses hidden_features)
        dropout_rate: Dropout probability
        act: Non-linear activation function, or None to disable the activation
            entirely (default: jax.nn.relu)
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', 'graph_norm', None).
            'graph_norm' is applied between GIN blocks; the MLP inside each
            GINConv uses 'layer_norm' instead, because GINConv does not plumb a
            batch vector into its MLP.
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        train_eps: Whether to learn the epsilon parameter
        rngs: Random number generators
    """

    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def init_conv(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> MessagePassing:
        """Initialize GINConv layer.

        Remaining keyword arguments (:obj:`eps`, :obj:`train_eps`) reach the
        :class:`GINConv` constructor, so an unsupported one raises a
        :obj:`TypeError` instead of being dropped.
        """
        # GINConv calls its MLP without a batch vector, so GraphNorm would pool
        # statistics over the whole disjoint union; use per-node LayerNorm there
        # and let GraphNorm act between the GIN blocks.
        mlp_norm: str | None
        if self.norm_type == "graph_norm":
            mlp_norm = "layer_norm"
        else:
            mlp_norm = self.norm_type

        # Create MLP for GINConv. Dropout is applied once per block by BasicGNN.
        mlp = MLP(
            feature_list=[in_features, out_features, out_features],
            act=self.act,
            act_first=self.act_first,
            norm=mlp_norm,
            rngs=rngs,
        )

        return GINConv(
            mlp,
            rngs=rngs,
            **kwargs,
        )
