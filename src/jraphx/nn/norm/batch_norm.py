import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.module import first_from
from flax.nnx.nn import initializers
from flax.typing import Dtype, Initializer


class BatchNorm(nnx.Module):
    r"""Applies batch normalization over a batch of node features as described in
    the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all nodes
    inside the mini-batch.

    Args:
        num_features (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): Decay rate for the exponential moving
            average of the batch statistics. Higher values mean slower adaptation
            (more weight on past values). (default: :obj:`0.99`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
        use_running_average (bool, optional): If set to :obj:`True`, use
            running statistics instead of batch statistics during evaluation.
            (default: :obj:`False`)
        dtype: The dtype of the result (default: infer from input and params).
        param_dtype: The dtype passed to parameter initializers (default: float32).
        use_bias (bool, optional): If True, bias (beta) is added.
            (default: :obj:`True`)
        use_scale (bool, optional): If True, multiply by scale (gamma).
            (default: :obj:`True`)
        bias_init: Initializer for bias, by default, zero.
        scale_init: Initializer for scale, by default, one.
        rngs: Random number generators for initialization.

    .. note::
        Statistics are always pooled over the node axis with features on the
        last axis, and there is no cross-device statistics synchronization;
        the ``axis``, ``axis_name``, ``axis_index_groups`` and
        ``use_fast_variance`` arguments of :class:`flax.nnx.BatchNorm` do not
        exist here.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.99,
        track_running_stats: bool = True,
        use_running_average: bool = False,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = initializers.zeros_init(),
        scale_init: Initializer = initializers.ones_init(),
        rngs: nnx.Rngs | None = None,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.use_running_average = use_running_average
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init

        feature_shape = (num_features,)

        # Learnable parameters
        self.weight: nnx.Param | None = nnx.data(None)
        self.bias: nnx.Param | None = nnx.data(None)

        if use_scale or use_bias:
            if rngs is not None:
                if use_scale:
                    key = rngs.params()
                    self.weight = nnx.Param(scale_init(key, feature_shape, param_dtype))
                if use_bias:
                    key = rngs.params()
                    self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
            else:
                # Fallback for backward compatibility when no rngs provided
                if use_scale:
                    self.weight = nnx.Param(jnp.ones(feature_shape))
                if use_bias:
                    self.bias = nnx.Param(jnp.zeros(feature_shape))

        # Running statistics
        self.running_mean: nnx.BatchStat | None
        self.running_var: nnx.BatchStat | None
        self.num_batches_tracked: nnx.BatchStat | None
        if track_running_stats:
            self.running_mean = nnx.BatchStat(jnp.zeros(num_features))
            self.running_var = nnx.BatchStat(jnp.ones(num_features))
            self.num_batches_tracked = nnx.BatchStat(jnp.array(0, dtype=jnp.int32))
        else:
            self.running_mean = nnx.data(None)
            self.running_var = nnx.data(None)
            self.num_batches_tracked = nnx.data(None)

    def __call__(
        self,
        x: jax.Array,
        batch: jax.Array | None = None,
        *,
        use_running_average: bool | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Apply batch normalization.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch assignment vector [num_nodes]. Accepted for API
                symmetry with :class:`~jraphx.nn.norm.GraphNorm` and
                :class:`~jraphx.nn.norm.LayerNorm`, and deliberately ignored:
                batch normalization pools statistics over every node of the
                mini-batch regardless of graph membership. Use
                :class:`~jraphx.nn.norm.GraphNorm` for per-graph statistics.
            use_running_average: If True, use running statistics. If False, compute
                batch statistics. If None, determined by training state.
            mask: Binary array for masked normalization (optional)

        Returns:
            Normalized features [num_nodes, num_features]
        """
        # Use Flax pattern to determine use_running_average
        use_running_average = first_from(
            use_running_average,
            self.use_running_average,
            error_msg="""No `use_running_average` argument was provided to BatchNorm
        as either a __call__ argument, class attribute, or nnx.flag.""",
        )

        if not use_running_average:
            # Statistics pooled over all nodes of the mini-batch
            if mask is not None:
                mean = jnp.average(x, axis=0, weights=mask)
                var = jnp.average((x - mean) ** 2, axis=0, weights=mask)
                count = jnp.sum(mask).astype(var.dtype)
            else:
                mean = x.mean(axis=0)
                var = x.var(axis=0)
                count = jnp.asarray(x.shape[0], dtype=var.dtype)

            # Update running statistics
            if (
                self.running_mean is not None
                and self.running_var is not None
                and self.num_batches_tracked is not None
            ):
                # Running variance tracks the unbiased estimator, matching
                # PyTorch/PyG, while normalization uses the biased one.
                unbiased_var = var * count / jnp.maximum(count - 1.0, 1.0)
                self.running_mean[...] = (
                    self.momentum * self.running_mean[...] + (1 - self.momentum) * mean
                )
                self.running_var[...] = (
                    self.momentum * self.running_var[...] + (1 - self.momentum) * unbiased_var
                )
                self.num_batches_tracked[...] += 1
        else:
            # Use running statistics
            if self.running_mean is not None and self.running_var is not None:
                mean = self.running_mean[...]
                var = self.running_var[...]
            else:
                # Fallback to batch statistics
                if mask is not None:
                    mean = jnp.average(x, axis=0, weights=mask)
                    var = jnp.average((x - mean) ** 2, axis=0, weights=mask)
                else:
                    mean = x.mean(axis=0)
                    var = x.var(axis=0)

        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        # Scale and shift
        out = x_norm
        if self.weight is not None:
            out = self.weight[...] * out
        if self.bias is not None:
            out = out + self.bias[...]

        # Apply dtype conversion if specified
        if self.dtype is not None:
            out = out.astype(self.dtype)

        return out
