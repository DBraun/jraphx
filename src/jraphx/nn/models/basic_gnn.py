"""Basic GNN base class for pre-built models."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from jraphx.nn.conv import MessagePassing
from jraphx.nn.models.jumping_knowledge import JumpingKnowledge
from jraphx.nn.norm import BatchNorm, GraphNorm, LayerNorm


class BasicGNN(nnx.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_features (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_features (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_features (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_features`. (default: :obj:`None`)
        dropout_rate (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to
            use. (default: :obj:`jax.nn.relu`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        norm (str, optional): The normalization function to
            use (:obj:`"batch_norm"`, :obj:`"layer_norm"`, :obj:`"graph_norm"`).
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`None`)
        residual (bool, optional): Whether to use residual connections between
            layers. (default: :obj:`False`)
        rngs: Random number generators for initialization.
        **kwargs (optional): Additional arguments for the specific convolution layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        out_features: int | None = None,
        dropout_rate: float = 0.0,
        act: Callable | None = None,
        act_first: bool = False,
        norm: str | None = None,
        jk: str | None = None,
        residual: bool = False,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.act = act if act is not None else nnx.relu
        self.act_first = act_first
        self.norm_type = norm
        self.jk_mode = jk
        self.residual = residual

        # Set output features
        if out_features is not None:
            self.out_features = out_features
        else:
            self.out_features = hidden_features

        # Create dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

        # Create convolution layers
        self.convs = nnx.List([])

        # First layer
        if num_layers > 0:
            self.convs.append(self.init_conv(in_features, hidden_features, rngs=rngs, **kwargs))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(hidden_features, hidden_features, rngs=rngs, **kwargs))

        # Last layer
        if num_layers >= 2:
            if out_features is not None and jk is None:
                # Output layer with different dimensions
                self.convs.append(
                    self.init_conv(hidden_features, out_features, rngs=rngs, **kwargs)
                )
            else:
                # Keep hidden dimensions
                self.convs.append(
                    self.init_conv(hidden_features, hidden_features, rngs=rngs, **kwargs)
                )

        # Create normalization layers
        self.norms = nnx.List([])
        for i in range(num_layers):
            # Determine the number of features for this layer
            if i == num_layers - 1 and out_features is not None and jk is None:
                norm_features = out_features
            else:
                norm_features = hidden_features

            if norm == "batch_norm":
                norm_layer = BatchNorm(norm_features, rngs=rngs)
            elif norm == "layer_norm":
                norm_layer = LayerNorm(norm_features)
            elif norm == "graph_norm":
                norm_layer = GraphNorm(norm_features)
            else:
                norm_layer = None
            self.norms.append(norm_layer)

        # Create JumpingKnowledge aggregation
        if jk is not None and jk != "last":
            self.jk = JumpingKnowledge(
                jk, num_features=hidden_features, num_layers=num_layers, rngs=rngs
            )
        else:
            self.jk = None

        # Output projection for JumpingKnowledge
        if jk is not None:
            if jk == "cat":
                jk_features = num_layers * hidden_features
            else:
                jk_features = hidden_features

            self.lin = nnx.Linear(jk_features, self.out_features, rngs=rngs)
        else:
            self.lin = None

    def init_conv(
        self, in_features: int, out_features: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> MessagePassing:
        """Initialize convolution layer. To be implemented by subclasses."""
        raise NotImplementedError

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
        batch: jnp.ndarray | None = None,
        batch_size: int | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch vector for batch normalization
            batch_size: Number of graphs in batch

        Returns:
            Output node features [num_nodes, out_features]
        """
        xs = []  # For JumpingKnowledge

        for i, conv in enumerate(self.convs):
            # Store input for residual connection
            if self.residual and i > 0:
                x_res = x

            # Convolution
            # Check what the conv layer supports
            if edge_attr is not None and hasattr(conv, "edge_dim"):
                x = conv(x, edge_index, edge_attr)
            elif edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)

            # Add residual connection (skip first layer)
            if self.residual and i > 0 and x_res.shape == x.shape:
                x = x + x_res

            # Apply normalization, activation, dropout (except possibly last layer)
            if i < self.num_layers - 1 or self.jk_mode is not None:
                # Activation first (if configured)
                if self.act is not None and self.act_first:
                    x = self.act(x)

                # Normalization
                if self.norms[i] is not None:
                    norm = self.norms[i]
                    if self.norm_type == "batch_norm" and batch is not None:
                        x = norm(x, batch)
                    elif self.norm_type == "graph_norm" and batch is not None:
                        x = norm(x, batch)
                    else:
                        x = norm(x)

                # Activation (if not first)
                if self.act is not None and not self.act_first:
                    x = self.act(x)

                # Dropout
                if self.dropout is not None:
                    x = self.dropout(x)

                # Store for JumpingKnowledge
                if self.jk is not None:
                    xs.append(x)

        # Apply JumpingKnowledge aggregation
        if self.jk is not None:
            x = self.jk(xs)

        # Final linear projection for JumpingKnowledge
        if self.lin is not None:
            x = self.lin(x)

        return x
