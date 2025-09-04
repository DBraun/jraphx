"""Scatter softmax operations for attention mechanisms in GNNs.

This module provides scatter-based softmax and log-softmax operations
that are essential for attention-based graph neural networks like GAT.
"""

from functools import partial

import jax
from jax import numpy as jnp

from .scatter import scatter_add, scatter_logsumexp, scatter_max


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_softmax(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Computes softmax over values grouped by index.

    For each group of values sharing the same index, computes:
    softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    This is commonly used in attention mechanisms where we need to
    normalize attention scores over neighboring nodes.

    Args:
        src: Source tensor with values to apply softmax to
        index: Indices determining which group each value belongs to
        dim_size: Number of groups (inferred if None)
        dim: Dimension along which to scatter (default: -2)
        temperature: Temperature parameter for softmax scaling

    Returns:
        Tensor with softmax applied within each group

    Example:
        >>> src = jnp.array([1.0, 2.0, 3.0, 1.5])
        >>> index = jnp.array([0, 0, 1, 1])
        >>> scatter_softmax(src, index, dim_size=2)
        # Group 0: softmax([1.0, 2.0]) = [0.27, 0.73]
        # Group 1: softmax([3.0, 1.5]) = [0.82, 0.18]
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        dim_size = index.max() + 1

    # Apply temperature scaling
    src = src / temperature

    # For numerical stability, subtract the max value per group
    max_vals = scatter_max(src, index, dim_size, dim)

    # Handle empty groups (where max is -inf)
    max_vals = jnp.where(jnp.isfinite(max_vals), max_vals, 0.0)

    # Subtract max from each element in its group
    src_shifted = src - max_vals[index]

    # Compute exp
    exp_vals = jnp.exp(src_shifted)

    # Sum exp values per group
    sum_exp = scatter_add(exp_vals, index, dim_size, dim)

    # Avoid division by zero for empty groups
    sum_exp = jnp.maximum(sum_exp, 1e-10)

    # Normalize: divide each exp value by its group's sum
    softmax_vals = exp_vals / sum_exp[index]

    return softmax_vals


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def scatter_log_softmax(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Computes log-softmax over values grouped by index.

    For each group of values sharing the same index, computes:
    log_softmax(x_i) = x_i - log(sum_j exp(x_j))

    This is numerically more stable than log(softmax(x)) and is useful
    for computing cross-entropy losses in attention mechanisms.

    Args:
        src: Source tensor with values to apply log-softmax to
        index: Indices determining which group each value belongs to
        dim_size: Number of groups (inferred if None)
        dim: Dimension along which to scatter (default: -2)
        temperature: Temperature parameter for softmax scaling

    Returns:
        Tensor with log-softmax applied within each group

    Example:
        >>> src = jnp.array([1.0, 2.0, 3.0, 1.5])
        >>> index = jnp.array([0, 0, 1, 1])
        >>> scatter_log_softmax(src, index, dim_size=2)
        # Group 0: log_softmax([1.0, 2.0])
        # Group 1: log_softmax([3.0, 1.5])
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    if dim_size is None:
        dim_size = index.max() + 1

    # Apply temperature scaling
    src = src / temperature

    # Use logsumexp for numerical stability
    logsumexp_vals = scatter_logsumexp(src, index, dim_size, dim)

    # Handle empty groups
    logsumexp_vals = jnp.where(jnp.isfinite(logsumexp_vals), logsumexp_vals, 0.0)

    # log_softmax = x - logsumexp(x)
    log_softmax_vals = src - logsumexp_vals[index]

    return log_softmax_vals


def masked_scatter_softmax(
    src: jnp.ndarray,
    index: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    dim_size: int | None = None,
    dim: int = -2,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Computes masked softmax over values grouped by index.

    Similar to scatter_softmax but with optional masking to exclude
    certain values from the softmax computation. Masked values are
    set to zero in the output.

    Args:
        src: Source tensor with values to apply softmax to
        index: Indices determining which group each value belongs to
        mask: Boolean mask, True for values to include (shape matching src)
        dim_size: Number of groups (inferred if None)
        dim: Dimension along which to scatter (default: -2)
        temperature: Temperature parameter for softmax scaling

    Returns:
        Tensor with masked softmax applied within each group
    """
    if mask is not None:
        # Set masked values to -inf before softmax
        src = jnp.where(mask, src, -jnp.inf)

    # Compute regular softmax
    result = scatter_softmax(src, index, dim_size, dim, temperature)

    # Set masked values to 0 in output
    if mask is not None:
        result = jnp.where(mask, result, 0.0)

    return result
