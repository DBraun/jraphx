"""Basic GNN base class for pre-built models."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from jraphx.nn.conv import MessagePassing
from jraphx.nn.models.jumping_knowledge import JumpingKnowledge
from jraphx.nn.norm import BatchNorm, GraphNorm, LayerNorm


class BasicGNN(nnx.Module):
    r"""An abstract class for implementing basic GNN models.

    Subclasses declare which optional edge information their convolution
    accepts through the :obj:`supports_edge_weight` and
    :obj:`supports_edge_attr` class attributes, both of which default to
    :obj:`False`. The forward pass only forwards an argument that the
    underlying convolution actually consumes, and raises a :obj:`ValueError`
    when it is handed edge information the subclass has not declared, so that a
    subclass missing those attributes fails loudly instead of silently training
    on an unweighted graph.

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
        act (Callable, optional): The non-linear activation function to use, or
            :obj:`None` to disable the activation entirely.
            (default: :obj:`jax.nn.relu`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        norm (str, optional): The normalization function to use
            (:obj:`"batch_norm"`, :obj:`"layer_norm"`, :obj:`"graph_norm"` or
            :obj:`None`). Any other value raises a :obj:`ValueError`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`None`)
        residual (bool, optional): Whether to use residual connections between
            layers. (default: :obj:`False`)
        rngs: Random number generators for initialization.
        **kwargs (optional): Additional arguments for the specific convolution layer.
    """

    supports_edge_weight: bool = False
    supports_edge_attr: bool = False

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        out_features: int | None = None,
        dropout_rate: float = 0.0,
        act: Callable | None = nnx.relu,
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
        self.act = act
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
        if num_layers >= 1:
            layer_in = in_features

            # First layer (only distinct from the last one for deeper networks)
            if num_layers > 1:
                self.convs.append(self.init_conv(layer_in, hidden_features, rngs=rngs, **kwargs))
                layer_in = hidden_features

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(self.init_conv(layer_in, hidden_features, rngs=rngs, **kwargs))

            # Last layer, which produces out_features unless JumpingKnowledge
            # aggregates the layer-wise representations instead
            if out_features is not None and jk is None:
                last_out = out_features
            else:
                last_out = hidden_features
            self.convs.append(self.init_conv(layer_in, last_out, rngs=rngs, **kwargs))

        # Validate the normalization choice once, so a typo cannot silently
        # disable normalization for the whole model
        if norm is not None and norm not in ("batch_norm", "layer_norm", "graph_norm"):
            raise ValueError(
                f"Unknown normalization {norm!r}; expected one of "
                "'batch_norm', 'layer_norm', 'graph_norm', or None"
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
            edge_weight: Edge weights [num_edges], requires
                :obj:`supports_edge_weight`
            edge_attr: Edge attributes [num_edges, edge_dim], requires
                :obj:`supports_edge_attr`
            batch: Batch vector used by ``batch_norm``, ``layer_norm`` and
                ``graph_norm``
            batch_size: Number of graphs in the mini-batch, forwarded to
                ``layer_norm`` and ``graph_norm``. Must be supplied as a Python
                :obj:`int` when the model is traced by :obj:`jax.jit`/
                :obj:`nnx.jit` together with a ``batch`` vector, since the
                number of segments is a static quantity.

        Returns:
            Output node features [num_nodes, out_features]

        Raises:
            ValueError: If ``edge_weight`` or ``edge_attr`` is given but the
                model does not declare support for it.
        """
        if edge_weight is not None and not self.supports_edge_weight:
            raise ValueError(
                f"'{type(self).__name__}' received 'edge_weight' but its convolution does "
                f"not consume edge weights. Set 'supports_edge_weight = True' on the "
                f"subclass if it does, or drop the argument"
            )
        if edge_attr is not None and not self.supports_edge_attr:
            raise ValueError(
                f"'{type(self).__name__}' received 'edge_attr' but its convolution does "
                f"not consume edge attributes. Set 'supports_edge_attr = True' on the "
                f"subclass if it does, or drop the argument"
            )

        xs = []  # For JumpingKnowledge

        for i, conv in enumerate(self.convs):
            # Store input for residual connection
            if self.residual:
                x_res = x

            # Convolution, forwarding only the edge information this model's
            # convolution actually consumes
            if self.supports_edge_weight and edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            elif self.supports_edge_attr and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            # Add residual connection whenever the widths line up
            if self.residual and x_res.shape == x.shape:
                x = x + x_res

            # Apply normalization, activation, dropout (except possibly last layer)
            if i < self.num_layers - 1 or self.jk_mode is not None:
                # Activation first (if configured)
                if self.act is not None and self.act_first:
                    x = self.act(x)

                # Normalization
                if self.norms[i] is not None:
                    norm = self.norms[i]
                    if self.norm_type == "batch_norm":
                        # BatchNorm pools over every node of the mini-batch and
                        # therefore has no segment count to make static
                        x = norm(x, batch)
                    else:
                        x = norm(x, batch, batch_size)

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
