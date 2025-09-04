import jax.numpy as jnp
from flax import nnx


class GraphNorm(nnx.Module):
    r"""Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean.

    Args:
        num_features (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        rngs: Random number generators for initialization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        rngs: nnx.Rngs | None = None,
    ):
        self.num_features = num_features
        self.eps = eps

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Learnable parameters
        self.weight = nnx.Param(jnp.ones(num_features))
        self.bias = nnx.Param(jnp.zeros(num_features))

    def __call__(
        self,
        x: jnp.ndarray,
        batch: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply graph normalization.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch assignment vector [num_nodes] (optional)

        Returns:
            Normalized features [num_nodes, num_features]
        """
        if batch is None:
            # Single graph case
            # Compute mean across all nodes and features
            mean = x.mean()
            # Subtract mean
            x = x - mean
            # Compute variance across all dimensions
            var = (x**2).mean()
            # Normalize
            x_norm = x / jnp.sqrt(var + self.eps)

        else:
            # Batched graph case
            batch_size = int(batch.max()) + 1
            x_norm = jnp.zeros_like(x)

            for b in range(batch_size):
                mask = batch == b
                graph_x = x[mask]

                if graph_x.shape[0] > 0:
                    # Compute mean across all nodes and features in this graph
                    mean = graph_x.mean()
                    # Subtract mean
                    graph_x = graph_x - mean
                    # Compute variance
                    var = (graph_x**2).mean()
                    # Normalize
                    graph_x = graph_x / jnp.sqrt(var + self.eps)
                    # Store result
                    x_norm = x_norm.at[mask].set(graph_x)

            x_norm = x_norm

        # Apply learnable scale and shift
        out = self.weight.value * x_norm + self.bias.value

        return out
