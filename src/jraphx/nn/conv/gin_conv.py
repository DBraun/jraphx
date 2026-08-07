"""Graph Isomorphism Network (GIN) layer implementation."""

from typing import Union

import jax
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
