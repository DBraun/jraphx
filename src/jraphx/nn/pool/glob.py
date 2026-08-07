"""Global pooling operations for graph-level representations.

This module provides pooling operations that:
1. Support an optional ``size`` parameter that makes the number of graphs static
2. Handle edge cases explicitly (single graph, no batch, empty graphs)
3. Use JAX segment operations directly for best performance

Under :func:`jax.jit` or :func:`jax.vmap` the number of graphs cannot be read off a
traced batch vector, so ``size`` must be passed explicitly there.
"""

from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp
from jax.ops import segment_max, segment_min, segment_sum


def _get_batch_size(batch: jax.Array | None, size: int | None = None) -> int:
    """Resolve the number of graphs as a static Python integer.

    Args:
        batch: Batch indices [num_nodes]
        size: Number of graphs, if known

    Returns:
        Number of graphs in batch

    Raises:
        ValueError: If ``size`` is not given and ``batch`` is a traced array, since the
            number of graphs would then depend on traced values.
    """
    if size is not None:
        return int(size)

    if batch is None:
        return 1

    if isinstance(batch, jax.core.Tracer):
        raise ValueError(
            "The number of graphs cannot be inferred from a traced batch vector. "
            "Pass `size=<num_graphs>` when pooling inside jax.jit or jax.vmap."
        )

    return int(batch.max()) + 1


def _align_per_graph(values: jax.Array, ndim: int) -> jax.Array:
    """Reshape a per-graph vector so it broadcasts against pooled features.

    Args:
        values: Per-graph values [batch_size]
        ndim: Rank of the pooled feature array to broadcast against

    Returns:
        ``values`` with ``ndim - 1`` trailing singleton axes appended
    """
    return values.reshape(values.shape + (1,) * (ndim - 1))


def _zero_empty_segments(pooled: jax.Array, batch: jax.Array, batch_size: int) -> jax.Array:
    """Replace the reduction identity of empty graphs by zeros.

    :func:`jax.ops.segment_max` and :func:`jax.ops.segment_min` fill segments without any
    node with :math:`\\mp\\infty`. Graph-level features of empty graphs are zero instead,
    which matches PyTorch Geometric and keeps downstream layers finite.

    Args:
        pooled: Graph-level features [batch_size, \\*feature_dims]
        batch: Batch indices for each node [num_nodes]
        batch_size: Number of graphs

    Returns:
        Graph-level features with empty graphs zeroed out
    """
    counts = segment_sum(
        jnp.ones(batch.shape[0], dtype=jnp.int32),
        batch,
        num_segments=batch_size,
    )
    mask = _align_per_graph(counts > 0, pooled.ndim)
    return jnp.where(mask, pooled, jnp.zeros_like(pooled))


def global_add_pool(
    x: jax.Array,
    batch: jax.Array | None = None,
    size: int | None = None,
) -> jax.Array:
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

    Args:
        x (jax.Array): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (jax.Array, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given, which requires ``batch`` to be a
            concrete (non-traced) array. (default: :obj:`None`)

    Returns:
        jax.Array: Graph-level features :math:`\mathbf{R} \in \mathbb{R}^{B \times F}`.
    """
    # Handle single graph case efficiently
    if batch is None:
        return x.sum(axis=0, keepdims=True)

    # Get batch size efficiently
    batch_size = _get_batch_size(batch, size)

    # Direct use of segment_sum for optimal performance
    return segment_sum(
        x,
        batch,
        num_segments=batch_size,
    )


def global_mean_pool(
    x: jax.Array,
    batch: jax.Array | None = None,
    size: int | None = None,
) -> jax.Array:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.

    Args:
        x (jax.Array): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (jax.Array, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given, which requires ``batch`` to be a
            concrete (non-traced) array. (default: :obj:`None`)

    Returns:
        jax.Array: Graph-level features :math:`\mathbf{R} \in \mathbb{R}^{B \times F}`.
    """
    # Handle single graph case efficiently
    if batch is None:
        return x.mean(axis=0, keepdims=True)

    # Get batch size efficiently
    batch_size = _get_batch_size(batch, size)

    # Compute sum using segment_sum
    sum_result = segment_sum(
        x,
        batch,
        num_segments=batch_size,
    )

    # Compute counts for each batch efficiently
    counts = segment_sum(
        jnp.ones(batch.shape[0]),
        batch,
        num_segments=batch_size,
    )

    # Avoid division by zero and compute mean
    counts = jnp.maximum(counts, 1.0)
    return sum_result / _align_per_graph(counts, sum_result.ndim)


def global_max_pool(
    x: jax.Array,
    batch: jax.Array | None = None,
    size: int | None = None,
) -> jax.Array:
    """Global max pooling over a batch of graphs.

    Computes the maximum of node features for each graph in the batch. Graphs without
    any node pool to zeros.

    Args:
        x: Node features [num_nodes, \\*feature_dims], with any number of feature axes
        batch: Batch indices for each node [num_nodes]
        size: Number of graphs in the batch (required under jit/vmap)

    Returns:
        Graph-level features [batch_size, \\*feature_dims]
    """
    # Handle single graph case efficiently
    if batch is None:
        return x.max(axis=0, keepdims=True)

    # Get batch size efficiently
    batch_size = _get_batch_size(batch, size)

    pooled = segment_max(
        x,
        batch,
        num_segments=batch_size,
    )
    return _zero_empty_segments(pooled, batch, batch_size)


def global_min_pool(
    x: jax.Array,
    batch: jax.Array | None = None,
    size: int | None = None,
) -> jax.Array:
    """Global min pooling over a batch of graphs.

    Computes the minimum of node features for each graph in the batch. Graphs without
    any node pool to zeros.

    Args:
        x: Node features [num_nodes, \\*feature_dims], with any number of feature axes
        batch: Batch indices for each node [num_nodes]
        size: Number of graphs in the batch (required under jit/vmap)

    Returns:
        Graph-level features [batch_size, \\*feature_dims]
    """
    # Handle single graph case efficiently
    if batch is None:
        return x.min(axis=0, keepdims=True)

    # Get batch size efficiently
    batch_size = _get_batch_size(batch, size)

    pooled = segment_min(
        x,
        batch,
        num_segments=batch_size,
    )
    return _zero_empty_segments(pooled, batch, batch_size)


def global_softmax_pool(
    x: jax.Array,
    batch: jax.Array | None = None,
    size: int | None = None,
    temperature: float = 1.0,
) -> jax.Array:
    """Global softmax pooling (weighted sum with softmax attention).

    Computes attention weights using softmax and performs weighted pooling.
    This is useful for differentiable pooling operations.

    Args:
        x: Node features [num_nodes, num_features]
        batch: Batch indices for each node [num_nodes]
        size: Number of graphs in the batch (required under jit/vmap)
        temperature: Temperature parameter for softmax

    Returns:
        Graph-level features [batch_size, num_features]
    """
    # Handle single graph case: attention is normalized over the nodes of the graph
    if batch is None:
        weights = nnx.softmax(x.sum(axis=-1) / temperature, axis=0).reshape(-1, 1)
        return (x * weights).sum(axis=0, keepdims=True)

    batch_size = _get_batch_size(batch, size)

    # Compute attention scores (sum across features)
    scores = x.sum(axis=-1) / temperature

    # Compute softmax per graph
    max_scores = segment_max(
        scores,
        batch,
        num_segments=batch_size,
    )

    # Numerically stable softmax
    scores = scores - jnp.take(max_scores, batch)
    exp_scores = jnp.exp(scores)

    # Sum of exponentials per graph
    sum_exp = segment_sum(
        exp_scores,
        batch,
        num_segments=batch_size,
    )

    # Compute weights
    weights = exp_scores / jnp.take(sum_exp, batch)

    # Weighted sum
    weighted_x = x * weights.reshape(-1, 1)
    return segment_sum(
        weighted_x,
        batch,
        num_segments=batch_size,
    )


def global_sort_pool(
    x: jax.Array,
    batch: jax.Array | None = None,
    k: int = 10,
    size: int | None = None,
) -> jax.Array:
    """Global sort pooling - select top-k features per graph.

    The SortPooling operator from `"An End-to-End Deep Learning Architecture
    for Graph Classification" <https://ojs.aaai.org/index.php/AAAI/article/view/11782>`_:
    nodes are sorted descending by their **last feature channel** (the paper's
    designated sort channel), the top ``k`` rows are kept, and graphs with fewer
    than ``k`` nodes are zero-padded *after* selection, so padding never
    displaces a real node regardless of the sign of its features.

    .. note::
        The batched path loops over graphs in Python and masks nodes by value, so it
        runs eagerly only.

    Args:
        x: Node features [num_nodes, num_features]
        batch: Batch indices for each node [num_nodes]
        k: Number of top nodes to select per graph
        size: Number of graphs in the batch

    Returns:
        Sorted and flattened features [batch_size, k * num_features]
    """

    def _sort_and_pad(graph_x: jax.Array) -> jax.Array:
        """Return the flattened top-k rows by last channel, zero-padded to k."""
        indices = jnp.argsort(-graph_x[:, -1])[:k]
        top = graph_x[indices]
        if top.shape[0] < k:
            padding = jnp.zeros((k - top.shape[0], graph_x.shape[1]), dtype=graph_x.dtype)
            top = jnp.concatenate([top, padding], axis=0)
        return top.flatten()

    if batch is None:
        # Single graph case
        return _sort_and_pad(x).reshape(1, -1)

    batch_size = _get_batch_size(batch, size)
    num_features = x.shape[1]

    # Initialize output
    output = jnp.zeros((batch_size, k * num_features))

    # Process each graph
    for i in range(batch_size):
        mask = batch == i
        graph_x = x[mask]

        if graph_x.shape[0] == 0:
            continue

        output = output.at[i].set(_sort_and_pad(graph_x))

    return output


def batch_histogram(
    x: jax.Array,
    batch: jax.Array | None = None,
    bins: int = 50,
    min_val: float | None = None,
    max_val: float | None = None,
    size: int | None = None,
) -> jax.Array:
    """Compute histogram features for each graph in batch.

    Creates fixed-size graph representations using histograms. Binning follows
    :func:`numpy.histogram`: bins are half-open except the last, which is
    closed on the right, and with an explicit ``min_val``/``max_val`` the
    values outside the range are dropped rather than folded into the edge
    bins. Bin edges are computed in the working precision (float32 by
    default), so a value lying exactly on an interior edge can land one bin
    away from :func:`numpy.histogram`, whose edges are float64.

    Args:
        x: Node features [num_nodes, num_features]
        batch: Batch indices for each node [num_nodes]
        bins: Number of histogram bins
        min_val: Minimum value for histogram
        max_val: Maximum value for histogram
        size: Number of graphs in the batch

    Returns:
        Histogram features [batch_size, bins * num_features]
    """
    # Determine value range
    lo = x.min() if min_val is None else min_val
    hi = x.max() if max_val is None else max_val

    # Handle single graph
    if batch is None:
        batch_size = 1
        batch = jnp.zeros(x.shape[0], dtype=jnp.int32)
    else:
        batch_size = _get_batch_size(batch, size)

    num_features = x.shape[1]
    output = jnp.zeros((batch_size, bins * num_features))

    # Compute bin edges
    bin_edges = jnp.linspace(lo, hi, bins + 1)

    # Process each feature and graph
    for feat_idx in range(num_features):
        feature_vals = x[:, feat_idx]

        # Digitize values. Searching the full edge array from the right and stepping
        # back one puts `v` in the bin whose half-open interval contains it, matching
        # :func:`numpy.histogram`; the clip folds `v == hi` into the last bin, which
        # is closed on the right.
        bin_indices = jnp.searchsorted(bin_edges, feature_vals, side="right") - 1
        bin_indices = jnp.clip(bin_indices, 0, bins - 1)

        # Values outside an explicit range are dropped, as numpy.histogram
        # drops them; weighting by the mask keeps every shape static for jit
        in_range = (feature_vals >= lo) & (feature_vals <= hi)

        # Create combined indices for 2D histogram
        combined_idx = batch * bins + bin_indices

        # Count occurrences in float32, so the tally stays exact for a low-precision
        # `x` whose accumulator would otherwise saturate
        hist = segment_sum(
            in_range.astype(jnp.float32),
            combined_idx,
            num_segments=batch_size * bins,
        ).reshape(batch_size, bins)

        # Store in output
        output = output.at[:, feat_idx * bins : (feat_idx + 1) * bins].set(hist)

    return output


# Batched versions for vmap compatibility. The batch vector is traced by vmap, so
# `size` is required: it is what makes the number of graphs static.
@partial(jax.vmap, in_axes=(0, 0, None))
def batched_global_add_pool(x: jax.Array, batch: jax.Array, size: int) -> jax.Array:
    """Map :func:`global_add_pool` over a leading batch-of-graphs axis.

    Args:
        x: Node features [num_batches, num_nodes, num_features]
        batch: Batch indices [num_batches, num_nodes]
        size: Number of graphs per entry, required because ``batch`` is traced

    Returns:
        Graph-level features [num_batches, size, num_features]
    """
    return global_add_pool(x, batch, size)


@partial(jax.vmap, in_axes=(0, 0, None))
def batched_global_mean_pool(x: jax.Array, batch: jax.Array, size: int) -> jax.Array:
    """Map :func:`global_mean_pool` over a leading batch-of-graphs axis.

    Args:
        x: Node features [num_batches, num_nodes, num_features]
        batch: Batch indices [num_batches, num_nodes]
        size: Number of graphs per entry, required because ``batch`` is traced

    Returns:
        Graph-level features [num_batches, size, num_features]
    """
    return global_mean_pool(x, batch, size)


@partial(jax.vmap, in_axes=(0, 0, None))
def batched_global_max_pool(x: jax.Array, batch: jax.Array, size: int) -> jax.Array:
    """Map :func:`global_max_pool` over a leading batch-of-graphs axis.

    Args:
        x: Node features [num_batches, num_nodes, num_features]
        batch: Batch indices [num_batches, num_nodes]
        size: Number of graphs per entry, required because ``batch`` is traced

    Returns:
        Graph-level features [num_batches, size, num_features]
    """
    return global_max_pool(x, batch, size)
