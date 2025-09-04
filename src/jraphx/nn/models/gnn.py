"""Pre-built GNN models (GCN, GAT, GraphSAGE, GIN)."""

from collections.abc import Callable

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
        act: Activation function
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', None)
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        improved: Use improved GCN normalization
        cached: Cache normalized edge weights for static graphs
        add_self_loops: Add self-loops to the graph
        normalize: Apply symmetric normalization
        rngs: Random number generators
    """

    def init_conv(
        self, in_features: int, out_features: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> MessagePassing:
        """Initialize GCNConv layer."""
        # Extract GCN-specific parameters
        improved = kwargs.pop("improved", False)
        cached = kwargs.pop("cached", False)
        add_self_loops = kwargs.pop("add_self_loops", True)
        normalize = kwargs.pop("normalize", True)

        return GCNConv(
            in_features,
            out_features,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=True,
            rngs=rngs,
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
        act: Activation function
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', None)
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        edge_dim: Edge feature dimension
        rngs: Random number generators
    """

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
        act: Callable | None = None,
        act_first: bool = False,
        norm: str | None = None,
        jk: str | None = None,
        residual: bool = False,
        edge_dim: int | None = None,
        rngs: nnx.Rngs | None = None,
        **kwargs,
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
        self, in_features: int, out_features: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> MessagePassing:
        """Initialize GATConv or GATv2Conv layer."""
        Conv = GATv2Conv if self.v2 else GATConv

        # For all but last layer, use multi-head with concat
        # For last layer (if no JK), use single output
        is_last = len(self.convs) == self.num_layers - 1
        use_concat = self.concat and not (is_last and self.jk_mode is None)

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
            edge_dim=self.edge_dim,
            residual=False,  # We handle residual in BasicGNN
            rngs=rngs,
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
        aggr: Aggregation method ('mean', 'max', 'lstm')
        dropout_rate: Dropout probability
        act: Activation function
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', None)
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        normalize: Whether to L2-normalize output features
        rngs: Random number generators
    """

    def init_conv(
        self, in_features: int, out_features: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> MessagePassing:
        """Initialize SAGEConv layer."""
        # Extract SAGE-specific parameters
        aggr = kwargs.pop("aggr", "mean")
        normalize = kwargs.pop("normalize", False)

        return SAGEConv(
            in_features,
            out_features,
            aggr=aggr,
            normalize=normalize,
            bias=True,
            rngs=rngs,
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
        act: Activation function
        act_first: If True, apply activation before normalization
        norm: Normalization type ('batch_norm', 'layer_norm', None)
        jk: Jumping Knowledge mode ('last', 'cat', 'max', 'lstm', None)
        residual: Whether to use residual connections
        train_eps: Whether to learn the epsilon parameter
        rngs: Random number generators
    """

    def init_conv(
        self, in_features: int, out_features: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> MessagePassing:
        """Initialize GINConv layer."""
        # Extract GIN-specific parameters
        train_eps = kwargs.pop("train_eps", False)

        # Create MLP for GINConv
        mlp = MLP(
            feature_list=[in_features, out_features, out_features],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm_type,
            dropout_rate=self.dropout_rate,
            rngs=rngs,
        )

        return GINConv(
            mlp,
            train_eps=train_eps,
            rngs=rngs,
        )
