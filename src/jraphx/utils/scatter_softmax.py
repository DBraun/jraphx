"""Scatter softmax operations for attention mechanisms in GNNs.

This module provides scatter-based softmax and log-softmax operations
that are essential for attention-based graph neural networks like GAT.
"""

from functools import partial

import jax
from jax import numpy as jnp

from .scatter import (
    _accumulator_dtype,
    _resolve_dim_size,
    scatter_add,
    scatter_logsumexp,
    scatter_max,
)


def scatter_softmax(
    src: jax.Array,
    index: jax.Array,
    dim_size: int | None = None,
    dim: int = -2,
    temperature: float = 1.0,
) -> jax.Array:
    """Computes softmax over values grouped by index.

    For each group of values sharing the same index, computes:
    softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    This is commonly used in attention mechanisms where we need to
    normalize attention scores over neighboring nodes.

    Groups whose entries are all :obj:`-inf` (the masked-attention idiom)
    produce zeros rather than NaN.

    Args:
        src: Source tensor with values to apply softmax to
        index: Indices determining which group each value belongs to
        dim_size: Number of groups, inferred from ``index`` if :obj:`None`
            (which requires a concrete ``index``)
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
    out: jax.Array = _scatter_softmax(
        src, index, _resolve_dim_size(index, dim_size), dim, temperature
    )
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def _scatter_softmax(
    src: jax.Array,
    index: jax.Array,
    dim_size: int,
    dim: int,
    temperature: float,
) -> jax.Array:
    """Jitted core of :func:`scatter_softmax` with a static ``dim_size``.

    Args:
        src: Source tensor with values to apply softmax to
        index: Indices determining which group each value belongs to
        dim_size: Number of groups
        dim: Dimension along which to scatter
        temperature: Temperature parameter for softmax scaling

    Returns:
        Tensor with softmax applied within each group
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    # Apply temperature scaling
    src = src / temperature

    # For numerical stability, subtract the max value per group
    max_vals = scatter_max(src, index, dim_size, dim)

    # A fully masked group has a max of -inf; shifting it by zero instead keeps
    # the exponentials finite and drives the group's output to zero.
    max_vals = jnp.where(jnp.isneginf(max_vals), jnp.zeros((), max_vals.dtype), max_vals)

    # Subtract max from each element in its group
    src_shifted = src - max_vals[index]

    # Compute exp, accumulating the per-group normalizer in at least float32: a
    # bfloat16 running sum freezes at 256, so the weights of any larger group
    # would sum to more than 1.
    exp_vals = jnp.exp(src_shifted)
    out_dtype = exp_vals.dtype
    exp_vals = exp_vals.astype(_accumulator_dtype(out_dtype))

    # Sum exp values per group
    sum_exp = scatter_add(exp_vals, index, dim_size, dim)

    # Empty and fully masked groups carry no mass; avoid dividing zero by zero
    sum_exp = jnp.where(sum_exp > 0, sum_exp, jnp.ones((), sum_exp.dtype))

    # Normalize: divide each exp value by its group's sum
    softmax_vals = (exp_vals / sum_exp[index]).astype(out_dtype)

    return softmax_vals


def scatter_log_softmax(
    src: jax.Array,
    index: jax.Array,
    dim_size: int | None = None,
    dim: int = -2,
    temperature: float = 1.0,
) -> jax.Array:
    """Computes log-softmax over values grouped by index.

    For each group of values sharing the same index, computes:
    log_softmax(x_i) = x_i - log(sum_j exp(x_j))

    This is numerically more stable than log(softmax(x)) and is useful
    for computing cross-entropy losses in attention mechanisms.

    Groups whose entries are all :obj:`-inf` (the masked-attention idiom)
    produce :obj:`-inf` rather than NaN, so exponentiating the result
    reproduces the zeros returned by :func:`scatter_softmax`.

    Args:
        src: Source tensor with values to apply log-softmax to
        index: Indices determining which group each value belongs to
        dim_size: Number of groups, inferred from ``index`` if :obj:`None`
            (which requires a concrete ``index``)
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
    out: jax.Array = _scatter_log_softmax(
        src, index, _resolve_dim_size(index, dim_size), dim, temperature
    )
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def _scatter_log_softmax(
    src: jax.Array,
    index: jax.Array,
    dim_size: int,
    dim: int,
    temperature: float,
) -> jax.Array:
    """Jitted core of :func:`scatter_log_softmax` with a static ``dim_size``.

    Args:
        src: Source tensor with values to apply log-softmax to
        index: Indices determining which group each value belongs to
        dim_size: Number of groups
        dim: Dimension along which to scatter
        temperature: Temperature parameter for softmax scaling

    Returns:
        Tensor with log-softmax applied within each group
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    # Apply temperature scaling
    src = src / temperature

    # Use logsumexp for numerical stability
    logsumexp_vals = scatter_logsumexp(src, index, dim_size, dim)

    # A fully masked group has a logsumexp of -inf, and every one of its entries
    # is -inf too; subtracting zero instead avoids -inf - (-inf) = NaN and leaves
    # the group at -inf, whose exponential is the zero produced by scatter_softmax.
    shift = jnp.where(
        jnp.isneginf(logsumexp_vals), jnp.zeros((), logsumexp_vals.dtype), logsumexp_vals
    )

    # log_softmax = x - logsumexp(x)
    log_softmax_vals = src - shift[index]

    return log_softmax_vals


def masked_scatter_softmax(
    src: jax.Array,
    index: jax.Array,
    mask: jax.Array | None = None,
    dim_size: int | None = None,
    dim: int = -2,
    temperature: float = 1.0,
) -> jax.Array:
    """Computes masked softmax over values grouped by index.

    Similar to scatter_softmax but with optional masking to exclude
    certain values from the softmax computation. Masked values are
    set to zero in the output.

    Args:
        src: Source tensor with values to apply softmax to
        index: Indices determining which group each value belongs to
        mask: Boolean mask, True for values to include (shape matching src)
        dim_size: Number of groups, inferred from ``index`` if :obj:`None`
            (which requires a concrete ``index``)
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
