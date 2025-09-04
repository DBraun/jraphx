"""Graph Isomorphism Network (GIN) layer implementation."""

import jax.numpy as jnp
from flax.nnx import Module, Param, Rngs

from jraphx.nn.conv.message_passing import MessagePassing


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
        if train_eps:
            self.eps = Param(jnp.array([eps]))
        else:
            self.eps = eps

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass of the GIN layer.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features (not used in GIN)

        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Get epsilon value
        if isinstance(self.eps, Param):
            eps = self.eps.value[0]
        else:
            eps = self.eps

        # Aggregate neighbor features
        out = self.propagate(edge_index, x, edge_attr)

        # Add weighted self-features
        out = (1 + eps) * x + out

        # Apply MLP
        out = self.nn(out)

        return out

    def message(
        self,
        x_j: jnp.ndarray,
        x_i: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Construct messages from source nodes.

        Args:
            x_j: Source node features [num_edges, in_features]
            x_i: Target node features (not used)
            edge_attr: Edge features (not used)

        Returns:
            Messages [num_edges, in_features]
        """
        return x_j
