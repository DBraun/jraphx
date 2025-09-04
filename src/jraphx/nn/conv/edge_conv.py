"""EdgeConv layer implementation for JraphX.

Based on "Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., 2019)
https://arxiv.org/abs/1801.07829
"""

from flax.nnx import Module
from jax import numpy as jnp

from .message_passing import MessagePassing


class EdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_features]` to shape :obj:`[-1, out_features]`,
            *e.g.*, defined by MLP.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
        self,
        nn: Module,
        aggr: str = "max",
    ):
        super().__init__(aggr=aggr)
        self.nn = nn

    def message(
        self,
        x_j: jnp.ndarray,
        x_i: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute messages using edge features (x_i, x_j - x_i)."""
        # Concatenate [x_i, x_j - x_i] and pass through network
        return self.nn(jnp.concatenate([x_i, x_j - x_i], axis=-1))

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        return self.propagate(edge_index, x)


class DynamicEdgeConv(Module):
    """Dynamic Edge Convolution layer with k-NN graph construction.

    This is a simplified version of PyTorch Geometric's DynamicEdgeConv that requires
    pre-computed k-NN indices. Unlike PyG's version which automatically computes
    k-nearest neighbors using torch-cluster, this implementation expects the k-NN
    indices to be provided as input.

    For true dynamic graph construction, you would need to:
    1. Compute k-NN indices from node features using a JAX k-NN implementation
    2. Pass these indices to this layer via the knn_indices parameter

    PyG equivalent: Uses torch_cluster.knn() for automatic k-NN computation.

    Args:
        nn: Neural network for edge features
        k: Number of nearest neighbors
        aggr: Aggregation method ('add', 'mean', 'max'). Default: 'max'
    """

    def __init__(
        self,
        nn: Module,
        k: int,
        aggr: str = "max",
    ):
        self.edge_conv = EdgeConv(nn, aggr=aggr)
        self.k = k

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray | None = None,
        knn_indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Optional pre-computed edge indices (ignored if knn_indices provided)
            knn_indices: Pre-computed k-NN indices [num_nodes, k]
                        (must be provided for dynamic graph construction)

        Returns:
            Updated node features [num_nodes, out_features]
        """
        if knn_indices is not None:
            # Convert k-NN indices to edge_index format
            num_nodes = x.shape[0]
            sources = jnp.repeat(jnp.arange(num_nodes), self.k)
            targets = knn_indices.flatten()
            edge_index = jnp.stack([sources, targets])
        elif edge_index is None:
            raise ValueError(
                "Either edge_index or knn_indices must be provided. "
                "For dynamic graph construction, pre-compute k-NN indices."
            )

        return self.edge_conv(x, edge_index)
