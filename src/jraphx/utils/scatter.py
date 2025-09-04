"""Scatter operations for aggregating values at indices using JAX.

This module provides optimized scatter operations using JAX's built-in
segment operations for better performance on GPU/TPU.
"""

from functools import partial

import jax
from jax import numpy as jnp


# Use optimized implementations by default
@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_add(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    r"""Sums all values from the :obj:`src` tensor at the indices specified
    in the :obj:`index` tensor along a given dimension ``dim``.

    Uses JAX's optimized segment_sum for better performance.

    .. note::
        For JIT compatibility, :obj:`dim_size` must be provided as a static
        integer. If :obj:`dim_size` is :obj:`None`, the function will
        compute it dynamically, which may fail under JIT compilation.

    Args:
        src (jax.Array): The source tensor.
        index (jax.Array): The index tensor.
        dim_size (int, optional): The size of the output tensor at dimension
            ``dim``. If set to :obj:`None`, will create a minimal-sized output
            tensor according to ``index.max() + 1``. For JIT compatibility,
            this should be provided as a static integer. (default: :obj:`None`)
        dim (int, optional): The dimension along which to index.
            (default: :obj:`-2`)

    Returns:
        jax.Array: Tensor with scattered values summed at each index.
    """
    if index.ndim != 1:
        raise ValueError(
            f"The `index` argument must be one-dimensional " f"(got {index.ndim} dimensions)"
        )

    if dim == -2:
        dim = 0  # Convert to 0 for segment operations
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        # For JIT compatibility, we compute dim_size outside of jitted context
        # This will work for eager execution but may fail in JIT
        dim_size = index.max() + 1 if index.size > 0 else 0

    return jax.ops.segment_sum(
        src,
        index,
        num_segments=dim_size,
    )


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_mean(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    """Scatter mean operation - averages values from src at indices specified by index.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension
        dim: Dimension along which to scatter

    Returns:
        Tensor with scattered values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        dim_size = index.max() + 1

    # Compute sum and count efficiently
    sums = jax.ops.segment_sum(src, index, num_segments=dim_size)
    ones = jnp.ones((src.shape[0],) + (1,) * (src.ndim - 1), dtype=src.dtype)
    counts = jax.ops.segment_sum(ones, index, num_segments=dim_size)

    # Avoid division by zero
    counts = jnp.maximum(counts, 1.0)
    return sums / counts


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_max(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    """Scatter max operation - takes maximum of values from src at indices specified by index.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension
        dim: Dimension along which to scatter

    Returns:
        Tensor with scattered values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        dim_size = index.max() + 1

    result = jax.ops.segment_max(
        src,
        index,
        num_segments=dim_size,
    )

    # Replace -inf with 0 for empty segments
    return jnp.where(jnp.isfinite(result), result, 0.0)


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_min(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    """Scatter min operation - takes minimum of values from src at indices specified by index.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension
        dim: Dimension along which to scatter

    Returns:
        Tensor with scattered values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        dim_size = index.max() + 1

    result = jax.ops.segment_min(
        src,
        index,
        num_segments=dim_size,
    )

    # Replace inf with 0 for empty segments
    return jnp.where(jnp.isfinite(result), result, 0.0)


@partial(jax.jit, static_argnames=("dim_size", "dim", "reduce"))
def scatter(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    reduce: str = "add",
) -> jnp.ndarray:
    """Generic scatter operation using JAX's optimized segment operations.

    This function scatters values from src tensor at indices specified by index tensor,
    applying the specified reduction operation. Uses JAX's built-in segment operations
    which are XLA-optimized for better performance on GPU/TPU.

    Args:
        src: Source tensor to scatter [\\*, N, \\*]
        index: Indices where to scatter [N] or same shape as src
        dim_size: Size of the output dimension (inferred if None)
        dim: Dimension along which to scatter (default: -2, which maps to 0)
        reduce: Reduction operation - "add", "mean", "max", "min"

    Returns:
        Output tensor with scattered values [\\*, dim_size, \\*]
    """
    if reduce == "add":
        return scatter_add(src, index, dim_size, dim)
    elif reduce == "mean":
        return scatter_mean(src, index, dim_size, dim)
    elif reduce == "max":
        return scatter_max(src, index, dim_size, dim)
    elif reduce == "min":
        return scatter_min(src, index, dim_size, dim)
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


def segment_sum(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    """Computes the sum along segments of a tensor.

    Args:
        data: Input tensor
        segment_ids: Segment indices for each element
        num_segments: Total number of segments

    Returns:
        Tensor with segmented sums
    """
    if num_segments is None:
        num_segments = segment_ids.max() + 1

    return jax.ops.segment_sum(data, segment_ids, num_segments)


def segment_mean(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    """Computes the mean along segments of a tensor.

    Args:
        data: Input tensor
        segment_ids: Segment indices for each element
        num_segments: Total number of segments

    Returns:
        Tensor with segmented means
    """
    if num_segments is None:
        num_segments = segment_ids.max() + 1

    # Compute sum and count
    sums = jax.ops.segment_sum(data, segment_ids, num_segments)
    counts = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)

    # Avoid division by zero
    counts = jnp.where(counts == 0, 1, counts)

    return sums / counts


def segment_max(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    """Computes the maximum along segments of a tensor.

    Args:
        data: Input tensor
        segment_ids: Segment indices for each element
        num_segments: Total number of segments

    Returns:
        Tensor with segmented maximums
    """
    if num_segments is None:
        num_segments = segment_ids.max() + 1

    return jax.ops.segment_max(data, segment_ids, num_segments)


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_std(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    """Scatter standard deviation - computes std of values at indices.

    Uses the formula: std = sqrt(E[X^2] - E[X]^2)

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension
        dim: Dimension along which to scatter

    Returns:
        Tensor with scattered standard deviations
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        dim_size = index.max() + 1

    # Compute mean and mean of squares
    mean = scatter_mean(src, index, dim_size, dim)

    # Compute E[X^2]
    src_squared = src * src
    mean_squared = scatter_mean(src_squared, index, dim_size, dim)

    # Compute variance: E[X^2] - E[X]^2
    variance = mean_squared - mean * mean

    # Handle numerical issues (variance should never be negative)
    variance = jnp.maximum(variance, 0.0)

    # Return standard deviation
    return jnp.sqrt(variance)


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_logsumexp(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    """Scatter logsumexp - numerically stable log-sum-exp aggregation.

    Computes log(sum(exp(x))) for values at each index, with numerical stability.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension
        dim: Dimension along which to scatter

    Returns:
        Tensor with log-sum-exp aggregated values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        dim_size = index.max() + 1

    # For numerical stability, subtract the max value per segment
    max_vals = scatter_max(src, index, dim_size, dim)

    # Replace -inf with a large negative number for empty segments
    max_vals = jnp.where(jnp.isfinite(max_vals), max_vals, -1e10)

    # Subtract max from each element (indexed by segment)
    src_shifted = src - max_vals[index]

    # Compute exp and sum
    exp_vals = jnp.exp(src_shifted)
    sum_exp = scatter_add(exp_vals, index, dim_size, dim)

    # Compute log and add back the max
    result = jnp.log(sum_exp + 1e-10) + max_vals

    # Handle empty segments (where max_vals was -inf)
    result = jnp.where(jnp.isfinite(max_vals), result, -jnp.inf)

    return result


# Keep fallback for compatibility
def scatter_fallback(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    reduce: str = "add",
) -> jnp.ndarray:
    """Fallback scatter implementation using loops (slower but supports all dimensions).

    This is the original implementation kept for compatibility and testing.
    Use the main scatter() function for better performance.
    """
    # Handle the common case for GNNs: dim=0
    if dim == 0 or (dim == -2 and src.ndim == 2):
        # Simple case: scatter along first dimension
        if dim_size is None:
            dim_size = index.max() + 1

        # Initialize output
        if src.ndim == 1:
            shape = (dim_size,)
        else:
            shape = (dim_size,) + src.shape[1:]

        if reduce == "add":
            out = jnp.zeros(shape, dtype=src.dtype)
            for i in range(src.shape[0]):
                out = out.at[index[i]].add(src[i])
        elif reduce == "mean":
            out = jnp.zeros(shape, dtype=src.dtype)
            count = jnp.zeros((dim_size,), dtype=jnp.float32)
            for i in range(src.shape[0]):
                out = out.at[index[i]].add(src[i])
                count = count.at[index[i]].add(1.0)
            # Avoid division by zero
            count = jnp.where(count == 0, 1.0, count)
            if src.ndim > 1:
                count = count.reshape(-1, *([1] * (src.ndim - 1)))
            out = out / count
        elif reduce == "max":
            out = jnp.full(shape, -jnp.inf, dtype=src.dtype)
            for i in range(src.shape[0]):
                out = out.at[index[i]].max(src[i])
            out = jnp.where(out == -jnp.inf, 0, out)
        elif reduce == "min":
            out = jnp.full(shape, jnp.inf, dtype=src.dtype)
            for i in range(src.shape[0]):
                out = out.at[index[i]].min(src[i])
            out = jnp.where(out == jnp.inf, 0, out)
        else:
            raise ValueError(f"Unknown reduce operation: {reduce}")

        return out

    # General case (less common in GNNs)
    raise NotImplementedError(f"Scatter along dimension {dim} is not yet implemented")
