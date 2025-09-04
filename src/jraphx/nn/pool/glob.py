"""Global pooling operations for graph-level representations.

This module provides efficient pooling operations that:
1. Cache batch size computations when possible
2. Support optional batch_size parameter to avoid repeated max() calls
3. Handle edge cases efficiently (single graph, no batch)
4. Use JAX segment operations directly for best performance
"""

from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp
from jax.ops import segment_max, segment_min, segment_sum


def _get_batch_size(batch: jnp.ndarray | None, size: int | None = None) -> int:
    """Efficiently compute batch size.

    Args:
        batch: Batch indices [num_nodes]
        size: Optional batch size

    Returns:
        Number of graphs in batch
    """
    if size is not None:
        return size

    if batch is None:
        return 1

    # Use JAX's max directly without int conversion until needed
    return batch.max() + 1


def global_add_pool(
    x: jnp.ndarray,
    batch: jnp.ndarray | None = None,
    size: int | None = None,
) -> jnp.ndarray:
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
            Automatically calculated if not given. (default: :obj:`None`)

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
    x: jnp.ndarray,
    batch: jnp.ndarray | None = None,
    size: int | None = None,
) -> jnp.ndarray:
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
            Automatically calculated if not given. (default: :obj:`None`)

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
    return sum_result / counts.reshape(-1, 1)


def global_max_pool(
    x: jnp.ndarray,
    batch: jnp.ndarray | None = None,
    size: int | None = None,
) -> jnp.ndarray:
    """Optimized global max pooling over a batch of graphs.

    Computes the maximum of node features for each graph in the batch.

    Args:
        x: Node features [num_nodes, num_features]
        batch: Batch indices for each node [num_nodes]
        size: Number of graphs in the batch (avoids computing max)

    Returns:
        Graph-level features [batch_size, num_features]
    """
    # Handle single graph case efficiently
    if batch is None:
        return x.max(axis=0, keepdims=True)

    # Get batch size efficiently
    batch_size = _get_batch_size(batch, size)

    # Direct use of segment_max for optimal performance
    return segment_max(
        x,
        batch,
        num_segments=batch_size,
    )


def global_min_pool(
    x: jnp.ndarray,
    batch: jnp.ndarray | None = None,
    size: int | None = None,
) -> jnp.ndarray:
    """Optimized global min pooling over a batch of graphs.

    Computes the minimum of node features for each graph in the batch.

    Args:
        x: Node features [num_nodes, num_features]
        batch: Batch indices for each node [num_nodes]
        size: Number of graphs in the batch (avoids computing max)

    Returns:
        Graph-level features [batch_size, num_features]
    """
    # Handle single graph case efficiently
    if batch is None:
        return x.min(axis=0, keepdims=True)

    # Get batch size efficiently
    batch_size = _get_batch_size(batch, size)

    # Direct use of segment_min for optimal performance
    return segment_min(
        x,
        batch,
        num_segments=batch_size,
    )


def global_softmax_pool(
    x: jnp.ndarray,
    batch: jnp.ndarray | None = None,
    size: int | None = None,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Global softmax pooling (weighted sum with softmax attention).

    Computes attention weights using softmax and performs weighted pooling.
    This is useful for differentiable pooling operations.

    Args:
        x: Node features [num_nodes, num_features]
        batch: Batch indices for each node [num_nodes]
        size: Number of graphs in the batch
        temperature: Temperature parameter for softmax

    Returns:
        Graph-level features [batch_size, num_features]
    """
    # Handle single graph case
    if batch is None:
        weights = nnx.softmax(x.sum(axis=-1, keepdims=True) / temperature)
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
    x: jnp.ndarray,
    batch: jnp.ndarray | None = None,
    k: int = 10,
    size: int | None = None,
) -> jnp.ndarray:
    """Global sort pooling - select top-k features per graph.

    Sorts node features and selects top-k nodes per graph.
    Useful for graph classification tasks.

    Args:
        x: Node features [num_nodes, num_features]
        batch: Batch indices for each node [num_nodes]
        k: Number of top nodes to select per graph
        size: Number of graphs in the batch

    Returns:
        Sorted and flattened features [batch_size, k * num_features]
    """
    if batch is None:
        # Single graph case
        num_nodes = x.shape[0]
        if num_nodes < k:
            # Pad with zeros if needed
            padding = jnp.zeros((k - num_nodes, x.shape[1]))
            x = jnp.concatenate([x, padding], axis=0)

        # Sort by feature sum and take top k
        scores = x.sum(axis=-1)
        indices = jnp.argsort(-scores)[:k]
        return x[indices].flatten().reshape(1, -1)

    batch_size = _get_batch_size(batch, size)
    num_features = x.shape[1]

    # Initialize output
    output = jnp.zeros((batch_size, k * num_features))

    # Process each graph
    for i in range(batch_size):
        mask = batch == i
        graph_x = x[mask]

        num_nodes = graph_x.shape[0]
        if num_nodes == 0:
            continue

        if num_nodes < k:
            # Pad with zeros
            padding = jnp.zeros((k - num_nodes, num_features))
            graph_x = jnp.concatenate([graph_x, padding], axis=0)

        # Sort by feature sum and take top k
        scores = graph_x.sum(axis=-1)
        indices = jnp.argsort(-scores)[:k]
        sorted_features = graph_x[indices].flatten()

        output = output.at[i].set(sorted_features)

    return output


def batch_histogram(
    x: jnp.ndarray,
    batch: jnp.ndarray | None = None,
    bins: int = 50,
    min_val: float | None = None,
    max_val: float | None = None,
    size: int | None = None,
) -> jnp.ndarray:
    """Compute histogram features for each graph in batch.

    Creates fixed-size graph representations using histograms.

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
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()

    # Handle single graph
    if batch is None:
        batch_size = 1
        batch = jnp.zeros(x.shape[0], dtype=jnp.int32)
    else:
        batch_size = _get_batch_size(batch, size)

    num_features = x.shape[1]
    output = jnp.zeros((batch_size, bins * num_features))

    # Compute bin edges
    bin_edges = jnp.linspace(min_val, max_val, bins + 1)

    # Process each feature and graph
    for feat_idx in range(num_features):
        feature_vals = x[:, feat_idx]

        # Digitize values
        bin_indices = jnp.searchsorted(bin_edges[:-1], feature_vals)
        bin_indices = jnp.clip(bin_indices, 0, bins - 1)

        # Create combined indices for 2D histogram
        combined_idx = batch * bins + bin_indices

        # Count occurrences
        hist = segment_sum(
            jnp.ones_like(feature_vals),
            combined_idx,
            num_segments=batch_size * bins,
        ).reshape(batch_size, bins)

        # Store in output
        output = output.at[:, feat_idx * bins : (feat_idx + 1) * bins].set(hist)

    return output


# Batched versions for vmap compatibility
@partial(jax.vmap, in_axes=(0, 0, None))
def batched_global_add_pool(x, batch, size):
    """Batched version of global_add_pool for vmap."""
    return global_add_pool(x, batch, size)


@partial(jax.vmap, in_axes=(0, 0, None))
def batched_global_mean_pool(x, batch, size):
    """Batched version of global_mean_pool for vmap."""
    return global_mean_pool(x, batch, size)


@partial(jax.vmap, in_axes=(0, 0, None))
def batched_global_max_pool(x, batch, size):
    """Batched version of global_max_pool for vmap."""
    return global_max_pool(x, batch, size)
