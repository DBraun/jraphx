import math
from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Initializer

from jraphx.utils.dtype import parse_dtype


class LayerNorm(nnx.Module):
    r"""Applies layer normalization over each individual example in a batch
    of node features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    In :obj:`"node"` mode the mean and standard-deviation are calculated across
    the node channels of each node separately. In :obj:`"graph"` mode they are
    calculated across all nodes and all node channels separately for each object
    in a mini-batch.

    Args:
        num_features (int or list): Size of each input sample, or list of
            dimensions to normalize.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        elementwise_affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode (str, optional): The normalization mode to use for layer
            normalization (:obj:`"graph"` or :obj:`"node"`). If :obj:`"graph"`
            is used, each graph will be considered as an element to be
            normalized. If `"node"` is used, each node will be considered as
            an element to be normalized. (default: :obj:`"node"`)
        dtype: The dtype of the result (default: infer from input and params).
            Strings such as ``"bfloat16"`` are resolved with
            :func:`~jraphx.utils.parse_dtype`.
        param_dtype: The dtype passed to parameter initializers (default: float32).
            Accepts the same strings.
        use_bias (bool, optional): If True, bias (beta) is added.
            (default: :obj:`True`)
        use_scale (bool, optional): If True, multiply by scale (gamma).
            (default: :obj:`True`)
        bias_init: Initializer for bias, by default, zero.
        scale_init: Initializer for scale, by default, one.
        rngs: Random number generators for initialization.

    .. note::
        The reduction axes follow from ``mode`` and there is no cross-device
        statistics synchronization; the ``reduction_axes``, ``feature_axes``,
        ``axis_name``, ``axis_index_groups`` and ``use_fast_variance``
        arguments of :class:`flax.nnx.LayerNorm` do not exist here.
    """

    def __init__(
        self,
        num_features: Union[int, list[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        mode: str = "node",
        *,
        dtype: str | type | jnp.dtype | None = None,
        param_dtype: str | type | jnp.dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = initializers.zeros_init(),
        scale_init: Initializer = initializers.ones_init(),
        rngs: nnx.Rngs | None = None,
    ):
        self.normalized_shape: tuple[int, ...]
        if isinstance(num_features, int):
            self.normalized_shape = (num_features,)
        else:
            self.normalized_shape = tuple(num_features)

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.mode = mode
        self.dtype = None if dtype is None else parse_dtype(dtype)
        self.param_dtype = parse_dtype(param_dtype)
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init

        # Learnable parameters - maintain backward compatibility with elementwise_affine
        self.weight: nnx.Param | None = nnx.data(None)
        self.bias: nnx.Param | None = nnx.data(None)

        if elementwise_affine and (use_bias or use_scale):
            if rngs is not None:
                if use_scale:
                    key = rngs.params()
                    self.weight = nnx.Param(
                        scale_init(key, self.normalized_shape, self.param_dtype)
                    )
                if use_bias:
                    key = rngs.params()
                    self.bias = nnx.Param(bias_init(key, self.normalized_shape, self.param_dtype))
            else:
                # Fallback for backward compatibility when no rngs provided
                if use_scale:
                    self.weight = nnx.Param(jnp.ones(self.normalized_shape))
                if use_bias:
                    self.bias = nnx.Param(jnp.zeros(self.normalized_shape))

    def __call__(
        self,
        x: jax.Array,
        batch: jax.Array | None = None,
        batch_size: int | None = None,
        *,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Apply layer normalization.

        Args:
            x: Node features [num_nodes, *normalized_shape]
            batch: Batch assignment vector [num_nodes]. Only used in
                :obj:`"graph"` mode; if :obj:`None`, all nodes are treated as
                belonging to a single graph.
            batch_size: Number of graphs in the mini-batch. Must be supplied as
                a Python :obj:`int` when :obj:`"graph"` mode is traced by
                :obj:`jax.jit`/:obj:`nnx.jit`, since the number of segments is a
                static quantity. When omitted it is derived from ``batch``,
                which forces a host synchronization and therefore only works
                outside of a trace.
            mask: Binary array for masked normalization. Accepted for API
                symmetry and currently unused, since masking interacts with the
                feature axes that layer normalization reduces over.

        Returns:
            Normalized features [num_nodes, *normalized_shape]

        Raises:
            ValueError: If ``mode`` is neither :obj:`"node"` nor :obj:`"graph"`.
        """
        if self.mode == "node":
            # Standard layer norm per node, reducing over the feature axis only.
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)

        elif self.mode == "graph":
            # Reduce over both the node axis and every feature axis, separately
            # for each graph in the mini-batch.
            if batch is None:
                batch = jnp.zeros(x.shape[0], dtype=jnp.int32)
                batch_size = 1
            elif batch_size is None:
                batch_size = int(batch.max()) + 1

            feature_axes = tuple(range(1, x.ndim))
            num_features = math.prod(x.shape[1:])
            broadcast_shape = (-1,) + (1,) * (x.ndim - 1)

            # Accumulated in at least float32, never in a narrower `x.dtype`:
            # bfloat16 has 8 mantissa bits, so its consecutive integers stop at 256
            # and both the element count and the running total freeze there, skewing
            # every graph larger than that by a size-dependent factor.
            accum_dtype = jnp.promote_types(x.dtype, jnp.float32)
            widened = x.astype(accum_dtype)
            counts = jax.ops.segment_sum(
                jnp.ones(batch.shape[0], dtype=accum_dtype), batch, num_segments=batch_size
            )
            # Elements per graph; empty graphs are clamped to avoid a 0/0 mean.
            denom = jnp.maximum(counts, 1.0) * num_features

            graph_mean = (
                jax.ops.segment_sum(widened.sum(axis=feature_axes), batch, num_segments=batch_size)
                / denom
            )
            mean = graph_mean[batch].reshape(broadcast_shape)

            centered = widened - mean
            graph_var = (
                jax.ops.segment_sum(
                    (centered**2).sum(axis=feature_axes), batch, num_segments=batch_size
                )
                / denom
            )
            var = graph_var[batch].reshape(broadcast_shape)

        else:
            raise ValueError(f"Unknown LayerNorm mode '{self.mode}' (expected 'node' or 'graph')")

        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        # Apply affine transformation
        if self.elementwise_affine:
            if self.weight is not None:
                x_norm = self.weight[...] * x_norm
            if self.bias is not None:
                x_norm = x_norm + self.bias[...]

        # Graph mode widens its accumulators, so hand the caller's own dtype back
        # unless an explicit output dtype was requested
        if self.dtype is not None:
            x_norm = x_norm.astype(self.dtype)
        else:
            x_norm = x_norm.astype(x.dtype)

        return x_norm
