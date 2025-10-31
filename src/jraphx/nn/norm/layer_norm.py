from typing import Any, Union

import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.typing import Axes, Dtype, Initializer


class LayerNorm(nnx.Module):
    r"""Applies layer normalization over each individual example in a batch
    of node features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated across all nodes and all
    node channels separately for each object in a mini-batch.

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
        param_dtype: The dtype passed to parameter initializers (default: float32).
        use_bias (bool, optional): If True, bias (beta) is added.
            (default: :obj:`True`)
        use_scale (bool, optional): If True, multiply by scale (gamma).
            (default: :obj:`True`)
        bias_init: Initializer for bias, by default, zero.
        scale_init: Initializer for scale, by default, one.
        reduction_axes: Axes for computing normalization statistics.
        feature_axes: Feature axes for learned bias and scaling.
        axis_name: The axis name used to combine batch statistics from multiple devices.
        axis_index_groups: Groups of axis indices within that named axis.
        use_fast_variance: If true, use faster, but less numerically stable variance calculation.
        rngs: Random number generators for initialization.
    """

    def __init__(
        self,
        num_features: Union[int, list[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        mode: str = "node",
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = initializers.zeros_init(),
        scale_init: Initializer = initializers.ones_init(),
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        axis_name: str | None = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        if isinstance(num_features, int):
            self.normalized_shape = (num_features,)
        else:
            self.normalized_shape = tuple(num_features)

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.mode = mode
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.bias_init = bias_init
        self.scale_init = scale_init
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance

        # Learnable parameters - maintain backward compatibility with elementwise_affine
        self.weight: nnx.Param = nnx.data(None)
        self.bias: nnx.Param = nnx.data(None)

        if elementwise_affine and (use_bias or use_scale):
            if rngs is not None:
                if use_scale:
                    key = rngs.params()
                    self.weight = nnx.Param(scale_init(key, self.normalized_shape, param_dtype))
                if use_bias:
                    key = rngs.params()
                    self.bias = nnx.Param(bias_init(key, self.normalized_shape, param_dtype))
            else:
                # Fallback for backward compatibility when no rngs provided
                if use_scale:
                    self.weight = nnx.Param(jnp.ones(self.normalized_shape))
                if use_bias:
                    self.bias = nnx.Param(jnp.zeros(self.normalized_shape))

    def __call__(
        self,
        x: jnp.ndarray,
        batch: jnp.ndarray | None = None,
        *,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply layer normalization.

        Args:
            x: Node features [num_nodes, *normalized_shape]
            batch: Batch assignment vector [num_nodes] (optional)
            mask: Binary array for masked normalization (optional)

        Returns:
            Normalized features [num_nodes, *normalized_shape]
        """
        if self.mode == "node":
            # Standard layer norm per node
            # Note: mask support for layer norm is complex as it affects feature dimensions
            # For now, we keep the standard behavior and ignore mask
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)

        elif self.mode == "graph" and batch is not None:
            # Normalize within each graph
            batch_size = int(batch.max()) + 1
            out = jnp.zeros_like(x)

            for b in range(batch_size):
                mask = batch == b
                graph_x = x[mask]

                if graph_x.shape[0] > 0:
                    # Compute statistics for this graph
                    # Note: mask support for graph mode layer norm is complex
                    # For now, we keep standard behavior
                    mean = graph_x.mean(axis=-1, keepdims=True)
                    var = graph_x.var(axis=-1, keepdims=True)

                    # Normalize
                    graph_x_norm = (graph_x - mean) / jnp.sqrt(var + self.eps)

                    # Apply affine transformation
                    if self.elementwise_affine:
                        graph_x_norm = self.weight.value * graph_x_norm + self.bias.value

                    # Store result
                    out = out.at[mask].set(graph_x_norm)

            return out

        else:
            # Fallback to node-wise normalization
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        # Apply affine transformation
        if self.elementwise_affine:
            if self.weight is not None:
                x_norm = self.weight.value * x_norm
            if self.bias is not None:
                x_norm = x_norm + self.bias.value

        # Apply dtype conversion if specified
        if self.dtype is not None:
            x_norm = x_norm.astype(self.dtype)

        return x_norm
