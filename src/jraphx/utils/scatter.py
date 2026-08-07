"""Scatter operations for aggregating values at indices using JAX.

This module provides optimized scatter operations using JAX's built-in
segment operations for better performance on GPU/TPU.
"""

from functools import partial

import jax
from jax import numpy as jnp

#: Accepted spellings of the supported reductions, mapped to canonical names.
_REDUCE_ALIASES = {"sum": "add", "add": "add", "mean": "mean", "max": "max", "min": "min"}


def _canonical_reduce(reduce: str) -> str:
    """Maps a reduction spelling onto the canonical JraphX name.

    Args:
        reduce: Name of the reduction. ``"sum"`` is accepted as an alias of
            ``"add"`` for :obj:`torch_geometric` compatibility.

    Returns:
        str: The canonical reduction name.

    Raises:
        ValueError: If ``reduce`` names an unsupported reduction.
    """
    if reduce not in _REDUCE_ALIASES:
        raise ValueError(
            f"Unknown reduce operation: {reduce!r} (expected one of {sorted(_REDUCE_ALIASES)})"
        )
    return _REDUCE_ALIASES[reduce]


def _accumulator_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Returns the dtype a summation over ``dtype`` should accumulate in.

    Reductions that divide by a member count are widened to at least float32.
    bfloat16 and float16 saturate far too early to total a real neighborhood:
    bfloat16 has 8 mantissa bits, so its consecutive integers stop at 256 and both
    the running total and the count freeze there, which skews the result by a
    degree-dependent factor. Integer dtypes are left alone so that exact integer
    arithmetic is not rounded through a float.

    Args:
        dtype: The dtype of the values being summed.

    Returns:
        The dtype to accumulate in.
    """
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.promote_types(dtype, jnp.float32)
    return dtype


def _resolve_dim_size(index: jnp.ndarray, dim_size: int | None) -> int:
    """Returns the number of output segments as a static Python integer.

    Args:
        index: The index tensor the scatter is grouped by.
        dim_size: The requested output size, or :obj:`None` to infer it from
            ``index``.

    Returns:
        int: The output size along the scattered dimension.

    Raises:
        jax.errors.ConcretizationTypeError: If ``dim_size`` is :obj:`None` and
            ``index`` is a tracer, because the inferred size is then
            data-dependent and cannot be a static shape.
    """
    if dim_size is not None:
        return dim_size
    if index.size == 0:
        return 0
    return int(index.max()) + 1


def _empty_segment_mask(index: jnp.ndarray, dim_size: int, ndim: int) -> jnp.ndarray:
    """Builds a mask selecting the segments that received at least one value.

    Args:
        index: The index tensor the scatter is grouped by.
        dim_size: The number of output segments.
        ndim: Rank of the source tensor, used to make the mask broadcastable.

    Returns:
        jax.Array: Boolean mask of shape ``(dim_size, 1, ..., 1)``.
    """
    counts = jax.ops.segment_sum(
        jnp.ones_like(index, dtype=jnp.int32), index, num_segments=dim_size
    )
    return (counts > 0).reshape((-1,) + (1,) * (ndim - 1))


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
        :obj:`dim_size` determines the output shape and is therefore a static
        argument. When it is :obj:`None` it is inferred from ``index``, which
        requires ``index`` to be concrete and hence is not available under
        :obj:`jax.jit`.

    Args:
        src (jax.Array): The source tensor.
        index (jax.Array): The index tensor.
        dim_size (int, optional): The size of the output tensor at dimension
            ``dim``. If set to :obj:`None`, will create a minimal-sized output
            tensor according to ``index.max() + 1``. (default: :obj:`None`)
        dim (int, optional): The dimension along which to index.
            (default: :obj:`-2`)

    Returns:
        jax.Array: Tensor with scattered values summed at each index.
    """
    out: jnp.ndarray = _scatter_add(src, index, _resolve_dim_size(index, dim_size), dim)
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def _scatter_add(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
    dim: int,
) -> jnp.ndarray:
    """Jitted core of :func:`scatter_add` with a static ``dim_size``.

    Args:
        src: The source tensor.
        index: The index tensor.
        dim_size: The number of output segments.
        dim: The dimension along which to index.

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

    return jax.ops.segment_sum(
        src,
        index,
        num_segments=dim_size,
    )


def scatter_mean(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    """Scatter mean operation - averages values from src at indices specified by index.

    Empty segments are filled with zero.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension, inferred from ``index`` if
            :obj:`None` (which requires a concrete ``index``)
        dim: Dimension along which to scatter

    Returns:
        Tensor with scattered values
    """
    out: jnp.ndarray = _scatter_mean(src, index, _resolve_dim_size(index, dim_size), dim)
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def _scatter_mean(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
    dim: int,
) -> jnp.ndarray:
    """Jitted core of :func:`scatter_mean` with a static ``dim_size``.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Number of output segments
        dim: Dimension along which to scatter

    Returns:
        Tensor with scattered values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    # Both the running total and the member count are accumulated in at least
    # float32, never in a narrower `src.dtype`. bfloat16 carries 8 mantissa bits, so
    # its integer sequence stops at 256 -- 256 + 1 rounds straight back to 256 --
    # and both accumulators saturate well before a high-degree node is finished:
    # the count freezes at 256 and the total stalls, leaving the quotient wrong by
    # a degree-dependent factor with no warning.
    accum_dtype = _accumulator_dtype(src.dtype)
    sums = jax.ops.segment_sum(src.astype(accum_dtype), index, num_segments=dim_size)
    ones = jnp.ones((src.shape[0],) + (1,) * (src.ndim - 1), dtype=accum_dtype)
    counts = jax.ops.segment_sum(ones, index, num_segments=dim_size)

    # Avoid division by zero
    counts = jnp.maximum(counts, 1.0)

    # A floating-point input gets its own precision back; an integer one keeps the
    # float result that `jnp.true_divide` would have produced.
    quotient = sums / counts
    if jnp.issubdtype(src.dtype, jnp.floating):
        quotient = quotient.astype(src.dtype)
    return quotient


def scatter_max(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    fill_value: float | None = None,
) -> jnp.ndarray:
    """Scatter max operation - takes maximum of values from src at indices specified by index.

    Segments that receive no value are filled with ``fill_value``. Emptiness is
    determined from ``index`` alone, so genuine infinities and NaNs present in
    ``src`` are propagated untouched and the output keeps ``src``'s dtype.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension, inferred from ``index`` if
            :obj:`None` (which requires a concrete ``index``)
        dim: Dimension along which to scatter
        fill_value: Value assigned to empty segments, cast to ``src.dtype``.
            :obj:`None` means zero.

    Returns:
        Tensor with scattered values
    """
    out: jnp.ndarray = _scatter_max(
        src, index, _resolve_dim_size(index, dim_size), dim, fill_value=fill_value
    )
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim", "fill_value"))
def _scatter_max(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
    dim: int,
    fill_value: float | None,
) -> jnp.ndarray:
    """Jitted core of :func:`scatter_max` with a static ``dim_size``.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Number of output segments
        dim: Dimension along which to scatter
        fill_value: Value assigned to empty segments, :obj:`None` meaning zero

    Returns:
        Tensor with scattered values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    result = jax.ops.segment_max(
        src,
        index,
        num_segments=dim_size,
    )

    fill = (
        jnp.zeros((), src.dtype) if fill_value is None else jnp.asarray(fill_value, dtype=src.dtype)
    )
    return jnp.where(_empty_segment_mask(index, dim_size, src.ndim), result, fill)


def scatter_min(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    fill_value: float | None = None,
) -> jnp.ndarray:
    """Scatter min operation - takes minimum of values from src at indices specified by index.

    Segments that receive no value are filled with ``fill_value``. Emptiness is
    determined from ``index`` alone, so genuine infinities and NaNs present in
    ``src`` are propagated untouched and the output keeps ``src``'s dtype.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension, inferred from ``index`` if
            :obj:`None` (which requires a concrete ``index``)
        dim: Dimension along which to scatter
        fill_value: Value assigned to empty segments, cast to ``src.dtype``.
            :obj:`None` means zero.

    Returns:
        Tensor with scattered values
    """
    out: jnp.ndarray = _scatter_min(
        src, index, _resolve_dim_size(index, dim_size), dim, fill_value=fill_value
    )
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim", "fill_value"))
def _scatter_min(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
    dim: int,
    fill_value: float | None,
) -> jnp.ndarray:
    """Jitted core of :func:`scatter_min` with a static ``dim_size``.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Number of output segments
        dim: Dimension along which to scatter
        fill_value: Value assigned to empty segments, :obj:`None` meaning zero

    Returns:
        Tensor with scattered values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    result = jax.ops.segment_min(
        src,
        index,
        num_segments=dim_size,
    )

    fill = (
        jnp.zeros((), src.dtype) if fill_value is None else jnp.asarray(fill_value, dtype=src.dtype)
    )
    return jnp.where(_empty_segment_mask(index, dim_size, src.ndim), result, fill)


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
        src: Source tensor to scatter [N, \\*]
        index: One-dimensional indices where to scatter [N], one per row of
            ``src``
        dim_size: Size of the output dimension, inferred from ``index`` if
            :obj:`None` (which requires a concrete ``index``)
        dim: Dimension along which to scatter (default: -2, which maps to 0)
        reduce: Reduction operation - "add" (alias "sum"), "mean", "max", "min"

    Returns:
        Output tensor with scattered values [\\*, dim_size, \\*]
    """
    reduce = _canonical_reduce(reduce)

    if reduce == "add":
        return scatter_add(src, index, dim_size, dim)
    elif reduce == "mean":
        return scatter_mean(src, index, dim_size, dim)
    elif reduce == "max":
        return scatter_max(src, index, dim_size, dim)
    elif reduce == "min":
        return scatter_min(src, index, dim_size, dim)
    else:
        raise RuntimeError(f"Canonical reduce operation {reduce!r} has no implementation")


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
    num_segments = _resolve_dim_size(segment_ids, num_segments)

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
    num_segments = _resolve_dim_size(segment_ids, num_segments)

    # Accumulated in at least float32 so that neither the total nor the count
    # saturates for low-precision `data`; see :func:`_scatter_mean`.
    accum_dtype = _accumulator_dtype(data.dtype)
    sums = jax.ops.segment_sum(data.astype(accum_dtype), segment_ids, num_segments)
    counts = jax.ops.segment_sum(jnp.ones_like(data, dtype=accum_dtype), segment_ids, num_segments)

    # Avoid division by zero
    counts = jnp.where(counts == 0, 1.0, counts)

    quotient = sums / counts
    if jnp.issubdtype(data.dtype, jnp.floating):
        quotient = quotient.astype(data.dtype)
    return quotient


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
    num_segments = _resolve_dim_size(segment_ids, num_segments)

    return jax.ops.segment_max(data, segment_ids, num_segments)


def scatter_std(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    unbiased: bool = True,
) -> jnp.ndarray:
    """Scatter standard deviation - computes std of values at indices.

    The deviations are accumulated around the per-segment mean, which avoids the
    catastrophic cancellation of the ``E[X^2] - E[X]^2`` form. Segments holding
    fewer than two values (and empty segments) get a standard deviation of zero.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension, inferred from ``index`` if
            :obj:`None` (which requires a concrete ``index``)
        dim: Dimension along which to scatter
        unbiased: Whether to apply Bessel's correction, *i.e.* divide by
            ``count - 1`` instead of ``count``

    Returns:
        Tensor with scattered standard deviations
    """
    out: jnp.ndarray = _scatter_std(
        src, index, _resolve_dim_size(index, dim_size), dim, unbiased=unbiased
    )
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim", "unbiased"))
def _scatter_std(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
    dim: int,
    unbiased: bool,
) -> jnp.ndarray:
    """Jitted core of :func:`scatter_std` with a static ``dim_size``.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Number of output segments
        dim: Dimension along which to scatter
        unbiased: Whether to apply Bessel's correction

    Returns:
        Tensor with scattered standard deviations
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    # Widened for the whole computation, for the reason given in
    # :func:`_scatter_mean`: the mean, the squared deviations and the member count
    # all saturate in a narrow float and would drag the deviation down with them.
    accum_dtype = _accumulator_dtype(src.dtype)
    widened = src.astype(accum_dtype)

    mean = _scatter_mean(widened, index, dim_size, dim)

    # Accumulate squared deviations around the segment mean.
    centered = widened - mean[index]
    sum_sq = _scatter_add(centered * centered, index, dim_size, dim)

    counts = jax.ops.segment_sum(
        jnp.ones_like(index, dtype=accum_dtype), index, num_segments=dim_size
    ).reshape((-1,) + (1,) * (src.ndim - 1))

    denominator = jnp.maximum(counts - 1.0 if unbiased else counts, 1.0)
    result = jnp.sqrt(sum_sq / denominator)
    if jnp.issubdtype(src.dtype, jnp.floating):
        result = result.astype(src.dtype)
    return result


def scatter_logsumexp(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
) -> jnp.ndarray:
    """Scatter logsumexp - numerically stable log-sum-exp aggregation.

    Computes log(sum(exp(x))) for values at each index, with numerical stability.
    Segments that receive no value evaluate to ``-inf``, the identity of the
    operation.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension, inferred from ``index`` if
            :obj:`None` (which requires a concrete ``index``)
        dim: Dimension along which to scatter

    Returns:
        Tensor with log-sum-exp aggregated values
    """
    out: jnp.ndarray = _scatter_logsumexp(src, index, _resolve_dim_size(index, dim_size), dim)
    return out


@partial(jax.jit, static_argnames=("dim_size", "dim"))
def _scatter_logsumexp(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
    dim: int,
) -> jnp.ndarray:
    """Jitted core of :func:`scatter_logsumexp` with a static ``dim_size``.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Number of output segments
        dim: Dimension along which to scatter

    Returns:
        Tensor with log-sum-exp aggregated values
    """
    if dim == -2:
        dim = 0
    if dim != 0:
        raise NotImplementedError("Optimized scatter only supports dim=0")

    # For numerical stability, subtract the max value per segment.
    max_vals = _scatter_max(src, index, dim_size, dim, fill_value=-jnp.inf)

    # Empty segments, and segments whose values are all -inf, carry a -inf max;
    # shifting those by zero keeps the exponentials finite and yields -inf.
    shift = jnp.where(jnp.isneginf(max_vals), jnp.zeros((), max_vals.dtype), max_vals)

    # Accumulate the exponentials in at least float32: a bfloat16 running sum
    # freezes at 256 and under-reports any larger segment.
    exp_vals = jnp.exp(src - shift[index])
    out_dtype = exp_vals.dtype
    exp_vals = exp_vals.astype(_accumulator_dtype(out_dtype))
    sum_exp = _scatter_add(exp_vals, index, dim_size, dim)

    return (jnp.log(sum_exp) + shift).astype(out_dtype)


# Keep fallback for compatibility
def scatter_fallback(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int | None = None,
    dim: int = -2,
    reduce: str = "add",
) -> jnp.ndarray:
    """Fallback scatter implementation using a Python loop over rows (slow).

    This implementation is kept for compatibility and testing.
    Use the main scatter() function for better performance.

    Only scattering along the leading axis is implemented: ``dim=0``, or the
    default ``dim=-2`` with a two-dimensional ``src``. Any other combination
    raises :obj:`NotImplementedError`.

    Args:
        src: Source tensor to scatter
        index: Indices where to scatter
        dim_size: Size of the output dimension, inferred from ``index`` if
            :obj:`None`
        dim: Dimension along which to scatter
        reduce: Reduction operation - "add" (alias "sum"), "mean", "max", "min"

    Returns:
        Output tensor with scattered values
    """
    reduce = _canonical_reduce(reduce)

    # Handle the common case for GNNs: dim=0
    if dim == 0 or (dim == -2 and src.ndim == 2):
        dim_size = _resolve_dim_size(index, dim_size)

        # Initialize output
        shape: tuple[int, ...]
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
        elif reduce in ("max", "min"):
            is_max = reduce == "max"
            extreme: int | float
            if jnp.issubdtype(src.dtype, jnp.integer):
                extreme = jnp.iinfo(src.dtype).min if is_max else jnp.iinfo(src.dtype).max
            else:
                extreme = -jnp.inf if is_max else jnp.inf
            out = jnp.full(shape, extreme, dtype=src.dtype)
            seen = jnp.zeros((dim_size,), dtype=jnp.bool_)
            for i in range(src.shape[0]):
                if is_max:
                    out = out.at[index[i]].max(src[i])
                else:
                    out = out.at[index[i]].min(src[i])
                seen = seen.at[index[i]].set(True)
            mask = seen.reshape((-1,) + (1,) * (src.ndim - 1))
            out = jnp.where(mask, out, jnp.zeros((), src.dtype))
        else:
            raise RuntimeError(f"Canonical reduce operation {reduce!r} has no implementation")

        return out

    # General case (less common in GNNs)
    raise NotImplementedError(f"Scatter along dimension {dim} is not yet implemented")
