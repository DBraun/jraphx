"""Graph Isomorphism Network (GIN) layer implementation."""

from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Linear, Module, Param, Rngs

from jraphx.nn.conv.message_passing import MessagePassing


def _leading_in_features(nn: Module) -> int:
    """Return the input width of a network's first parameterized layer.

    Args:
        nn: The network whose input width is wanted. A module exposing
            ``in_features`` directly (:class:`~jraphx.nn.models.MLP`,
            :class:`flax.nnx.Linear`) is read as-is; for a
            :class:`flax.nnx.Sequential` the first layer is inspected.

    Returns:
        The number of input features.

    Raises:
        ValueError: If the input width cannot be inferred. Pass a network that
            exposes ``in_features``, or project the edge features yourself.
    """
    first: object = nn
    if isinstance(nn, nnx.Sequential):
        first = nn.layers[0]
    in_features = getattr(first, "in_features", None)
    if not isinstance(in_features, int):
        raise ValueError(
            f"Could not infer the input width of {type(first).__name__!r}; it exposes "
            f"no integer `in_features` attribute"
        )
    return in_features


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_features]` to
            shape :obj:`[-1, out_features]`, *e.g.*, defined by MLP.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        rngs: Random number generators for initialization.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
        self,
        nn: Module,
        eps: float = 0.0,
        train_eps: bool = False,
        rngs: Rngs | None = None,
    ):
        """Initialize the GIN layer."""
        super().__init__(aggr="add")

        self.nn = nn
        self.initial_eps = eps

        # Make epsilon learnable if requested
        self.eps: Param | float
        if train_eps:
            self.eps = Param(jnp.array([eps]))
        else:
            self.eps = eps

    def __call__(
        self,
        x: Union[jax.Array, tuple[jax.Array, jax.Array]],
        edge_index: jax.Array,
        edge_attr: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass of the GIN layer.

        Args:
            x: Node features [num_nodes, in_features], or a ``(x_src, x_dst)``
                tuple for bipartite graphs
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features (not used in GIN)

        Returns:
            Updated node features [num_nodes, out_features], or
            [num_dst_nodes, out_features] for bipartite input
        """
        # Get epsilon value
        eps: jax.Array | float
        if isinstance(self.eps, Param):
            eps = self.eps[0]
        else:
            eps = self.eps

        # The root term uses the target side of a bipartite pair
        if isinstance(x, tuple):
            x_dst = x[1] if self.flow == "source_to_target" else x[0]
        else:
            x_dst = x

        # Aggregate neighbor features
        out = self.propagate(edge_index, x, edge_attr)

        # Add weighted self-features
        out = (1 + eps) * x_dst + out

        # Apply MLP
        out_features: jax.Array = self.nn(out)

        return out_features

    def message(
        self,
        x_j: jax.Array,
        x_i: jax.Array | None = None,
        edge_attr: jax.Array | None = None,
    ) -> jax.Array:
        """Construct messages from source nodes.

        Args:
            x_j: Source node features [num_edges, in_features]
            x_i: Target node features (not used)
            edge_attr: Edge features (not used)

        Returns:
            Messages [num_edges, in_features]
        """
        return x_j


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_features]` to
            shape :obj:`[-1, out_features]`, *e.g.*, defined by
            :class:`~jraphx.nn.models.MLP`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Otherwise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        rngs: Random number generators for initialization. Required when
            ``edge_dim`` is set, since the edge projection draws parameters.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
        self,
        nn: Module,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: int | None = None,
        rngs: Rngs | None = None,
    ):
        """Initialize the GINE layer."""
        super().__init__(aggr="add")

        self.nn = nn
        self.initial_eps = eps
        self.edge_dim = edge_dim

        # Make epsilon learnable if requested
        self.eps: Param | float
        if train_eps:
            self.eps = Param(jnp.array([eps]))
        else:
            self.eps = eps

        # Edge features whose width differs from the node width must be
        # projected; the projection's input width is read off the wrapped
        # network's first layer
        self.lin: Linear | None
        if edge_dim is not None:
            if rngs is None:
                raise ValueError(
                    "'rngs' is required when 'edge_dim' is set, because the edge "
                    "projection draws parameters"
                )
            in_features = _leading_in_features(nn)
            self.lin = Linear(edge_dim, in_features, rngs=rngs)
        else:
            self.lin = nnx.data(None)

    def __call__(
        self,
        x: Union[jax.Array, tuple[jax.Array, jax.Array]],
        edge_index: jax.Array,
        edge_attr: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass of the GINE layer.

        Args:
            x: Node features [num_nodes, in_features], or a ``(x_src, x_dst)``
                tuple for bipartite graphs
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]. Required: every
                message adds the edge's features to the source features.

        Returns:
            Updated node features [num_nodes, out_features], or
            [num_dst_nodes, out_features] for bipartite input

        Raises:
            RuntimeError: If ``edge_attr`` is :obj:`None`.
            ValueError: If no ``edge_dim`` was given and the edge feature width
                differs from the node feature width.
        """
        if edge_attr is None:
            raise RuntimeError(
                "GINEConv requires edge_attr: every message adds the edge's features "
                "to the source node's features."
            )

        # Get epsilon value
        eps: jax.Array | float
        if isinstance(self.eps, Param):
            eps = self.eps[0]
        else:
            eps = self.eps

        # The root term uses the target side of a bipartite pair
        if isinstance(x, tuple):
            x_dst = x[1] if self.flow == "source_to_target" else x[0]
        else:
            x_dst = x

        # Aggregate neighbor features
        out = self.propagate(edge_index, x, edge_attr)

        # Add weighted self-features
        out = (1 + eps) * x_dst + out

        # Apply MLP
        out_features: jax.Array = self.nn(out)

        return out_features

    def message(
        self,
        x_j: jax.Array,
        x_i: jax.Array | None = None,
        edge_attr: jax.Array | None = None,
    ) -> jax.Array:
        """Construct messages by fusing edge features into source features.

        Args:
            x_j: Source node features [num_edges, in_features]
            x_i: Target node features (not used)
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Messages [num_edges, in_features]

        Raises:
            ValueError: If no edge projection exists and the edge feature width
                differs from the node feature width; broadcasting the narrower
                one would silently compute something other than
                :math:`\\mathbf{x}_j + \\mathbf{e}_{j,i}`.
        """
        assert edge_attr is not None  # guaranteed by __call__
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        elif edge_attr.shape[-1] != x_j.shape[-1]:
            raise ValueError(
                f"Edge features have width {edge_attr.shape[-1]} but node features "
                f"have width {x_j.shape[-1]}. Set 'edge_dim' so that GINEConv "
                f"projects the edge features to the node width."
            )
        return nnx.relu(x_j + edge_attr)
