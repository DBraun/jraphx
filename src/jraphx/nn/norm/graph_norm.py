import jax
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

    The mean and standard-deviation are computed per feature channel over the
    nodes of each individual graph in the mini-batch.

    Args:
        num_features (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        rngs: Random number generators for initialization. Accepted for API
            symmetry with the other normalization layers; the parameters use
            constant initializers and consume no randomness.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        rngs: nnx.Rngs | None = None,
    ):
        self.num_features = num_features
        self.eps = eps

        # Learnable parameters
        self.weight = nnx.Param(jnp.ones(num_features))
        self.bias = nnx.Param(jnp.zeros(num_features))
        self.mean_scale = nnx.Param(jnp.ones(num_features))

    def __call__(
        self,
        x: jnp.ndarray,
        batch: jnp.ndarray | None = None,
        batch_size: int | None = None,
    ) -> jnp.ndarray:
        """Apply graph normalization.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch assignment vector [num_nodes]. If :obj:`None`, all
                nodes are treated as belonging to a single graph.
            batch_size: Number of graphs in the mini-batch. Must be supplied as
                a Python :obj:`int` when this layer is traced by
                :obj:`jax.jit`/:obj:`nnx.jit`, since the number of segments is a
                static quantity. When omitted it is derived from ``batch``,
                which forces a host synchronization and therefore only works
                outside of a trace.

        Returns:
            Normalized features [num_nodes, num_features]
        """
        if batch is None:
            batch = jnp.zeros(x.shape[0], dtype=jnp.int32)
            batch_size = 1
        elif batch_size is None:
            batch_size = int(batch.max()) + 1

        # Per-graph statistics are accumulated in at least float32, never in a
        # narrower `x.dtype`: bfloat16 has 8 mantissa bits, so its consecutive
        # integers stop at 256 and both the node count and the running total freeze
        # there, skewing every graph with more nodes than that by a size-dependent
        # factor. Empty graphs are clamped to avoid a 0/0 mean.
        accum_dtype = jnp.promote_types(x.dtype, jnp.float32)
        widened = x.astype(accum_dtype)
        counts = jax.ops.segment_sum(
            jnp.ones(batch.shape[0], dtype=accum_dtype), batch, num_segments=batch_size
        )
        counts = jnp.maximum(counts, 1.0)[:, None]

        mean = jax.ops.segment_sum(widened, batch, num_segments=batch_size) / counts
        out = widened - self.mean_scale[...] * mean[batch]
        var = jax.ops.segment_sum(out**2, batch, num_segments=batch_size) / counts
        std = jnp.sqrt(var[batch] + self.eps)

        normalized = self.weight[...] * out / std + self.bias[...]
        return normalized.astype(x.dtype)
