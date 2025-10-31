from typing import Any

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
        axis (int, optional): The feature or non-batch axis of the input.
            (default: :obj:`-1`)
        dtype: The dtype of the result (default: infer from input and params).
        param_dtype: The dtype passed to parameter initializers (default: float32).
        use_bias (bool, optional): If True, bias (beta) is added.
            (default: :obj:`True`)
        use_scale (bool, optional): If True, multiply by scale (gamma).
            (default: :obj:`True`)
        bias_init: Initializer for bias, by default, zero.
        scale_init: Initializer for scale, by default, one.
        axis_name: The axis name used to combine batch statistics from multiple devices.
        axis_index_groups: Groups of axis indices within that named axis.
        use_fast_variance: If true, use faster, but less numerically stable variance calculation.
        rngs: Random number generators for initialization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.99,
        track_running_stats: bool = True,
        use_running_average: bool = False,
        *,
        axis: int = -1,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = initializers.zeros_init(),
        scale_init: Initializer = initializers.ones_init(),
        axis_name: str | None = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.use_running_average = use_running_average
        self.axis = axis
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance

        feature_shape = (num_features,)

        # Learnable parameters
        self.weight: nnx.Param = nnx.data(None)
        self.bias: nnx.Param = nnx.data(None)

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
        if track_running_stats:
            self.running_mean = nnx.Variable(jnp.zeros(num_features))
            self.running_var = nnx.Variable(jnp.ones(num_features))
            self.num_batches_tracked = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        else:
            self.running_mean = nnx.data(None)
            self.running_var = nnx.data(None)
            self.num_batches_tracked = nnx.data(None)

    def __call__(
        self,
        x: jnp.ndarray,
        batch: jnp.ndarray | None = None,
        *,
        use_running_average: bool | None = None,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply batch normalization.

        Args:
            x: Node features [num_nodes, num_features]
            batch: Batch assignment vector [num_nodes] (optional)
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
            # Compute batch statistics
            if batch is not None:
                # Compute per-batch statistics and average them
                batch_size = int(batch.max()) + 1
                mean = jnp.zeros((batch_size, self.num_features))
                var = jnp.zeros((batch_size, self.num_features))

                for b in range(batch_size):
                    batch_mask = batch == b
                    batch_x = x[batch_mask]
                    if batch_x.shape[0] > 0:
                        if mask is not None:
                            node_mask = mask[batch_mask]
                            mean = mean.at[b].set(jnp.average(batch_x, axis=0, weights=node_mask))
                            var = var.at[b].set(
                                jnp.average((batch_x - mean[b]) ** 2, axis=0, weights=node_mask)
                            )
                        else:
                            mean = mean.at[b].set(batch_x.mean(axis=0))
                            var = var.at[b].set(batch_x.var(axis=0))

                # Average across batches
                mean = mean.mean(axis=0)
                var = var.mean(axis=0)
            else:
                # Global statistics across all nodes
                if mask is not None:
                    mean = jnp.average(x, axis=0, weights=mask)
                    var = jnp.average((x - mean) ** 2, axis=0, weights=mask)
                else:
                    mean = x.mean(axis=0)
                    var = x.var(axis=0)

            # Update running statistics
            if self.track_running_stats:
                self.running_mean.value = (
                    self.momentum * self.running_mean.value + (1 - self.momentum) * mean
                )
                self.running_var.value = (
                    self.momentum * self.running_var.value + (1 - self.momentum) * var
                )
                self.num_batches_tracked.value += 1
        else:
            # Use running statistics
            if self.track_running_stats:
                mean = self.running_mean.value
                var = self.running_var.value
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
            out = self.weight.value * out
        if self.bias is not None:
            out = out + self.bias.value

        # Apply dtype conversion if specified
        if self.dtype is not None:
            out = out.astype(self.dtype)

        return out
