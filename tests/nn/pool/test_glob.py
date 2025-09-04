"""Test global pooling operations for JraphX.

Converted from PyTorch Geometric test_glob.py to test JraphX functionality.
"""

import jax.numpy as jnp
import pytest

from jraphx.nn.pool.glob import global_add_pool, global_max_pool, global_mean_pool


def test_global_pool():
    """Test basic global pooling operations."""
    # N_1, N_2 = 4, 6  # Not used directly in test
    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],  # First graph (4 nodes)
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
            [7.0, 8.0, 9.0, 10.0],
            [8.0, 9.0, 10.0, 11.0],
            [9.0, 10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )  # Second graph (6 nodes)

    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Test global_add_pool
    out = global_add_pool(x, batch)
    assert out.shape == (2, 4)
    assert jnp.allclose(out[0], x[:4].sum(axis=0))
    assert jnp.allclose(out[1], x[4:].sum(axis=0))

    # Test global_add_pool without batch (single graph)
    out = global_add_pool(x, None)
    assert out.shape == (1, 4)
    assert jnp.allclose(out, x.sum(axis=0, keepdims=True))

    # Test global_mean_pool
    out = global_mean_pool(x, batch)
    assert out.shape == (2, 4)
    assert jnp.allclose(out[0], x[:4].mean(axis=0))
    assert jnp.allclose(out[1], x[4:].mean(axis=0))

    # Test global_mean_pool without batch (single graph)
    out = global_mean_pool(x, None)
    assert out.shape == (1, 4)
    assert jnp.allclose(out, x.mean(axis=0, keepdims=True))

    # Test global_max_pool
    out = global_max_pool(x, batch)
    assert out.shape == (2, 4)
    assert jnp.allclose(out[0], x[:4].max(axis=0))
    assert jnp.allclose(out[1], x[4:].max(axis=0))

    # Test global_max_pool without batch (single graph)
    out = global_max_pool(x, None)
    assert out.shape == (1, 4)
    assert jnp.allclose(out, x.max(axis=0, keepdims=True))


def test_permuted_global_pool():
    """Test global pooling with permuted batch indices."""
    # N_1, N_2 = 4, 6  # Not used directly in test
    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],  # Should belong to batch 0
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
            [7.0, 8.0, 9.0, 10.0],
            [8.0, 9.0, 10.0, 11.0],
            [9.0, 10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )  # Should belong to batch 1

    # Define the graph sizes
    N_1, N_2 = 4, 6
    batch = jnp.concatenate([jnp.zeros(N_1, dtype=jnp.int32), jnp.ones(N_2, dtype=jnp.int32)])

    # Create permutation
    perm = jnp.array([8, 2, 5, 0, 9, 3, 1, 7, 4, 6])  # Random permutation

    px = x[perm]
    pbatch = batch[perm]
    px1 = px[pbatch == 0]
    px2 = px[pbatch == 1]

    # Test global_add_pool with permutation
    out = global_add_pool(px, pbatch)
    assert out.shape == (2, 4)
    assert jnp.allclose(out[0], px1.sum(axis=0))
    assert jnp.allclose(out[1], px2.sum(axis=0))

    # Test global_mean_pool with permutation
    out = global_mean_pool(px, pbatch)
    assert out.shape == (2, 4)
    assert jnp.allclose(out[0], px1.mean(axis=0))
    assert jnp.allclose(out[1], px2.mean(axis=0))

    # Test global_max_pool with permutation
    out = global_max_pool(px, pbatch)
    assert out.shape == (2, 4)
    assert jnp.allclose(out[0], px1.max(axis=0))
    assert jnp.allclose(out[1], px2.max(axis=0))


def test_dense_global_pool():
    """Test global pooling with dense (3D) input."""
    x = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
        ]
    )  # Shape: (3, 3, 2)

    # For dense input with no batch, global_add_pool sums along axis 0 (keeping dims)
    result = global_add_pool(x, None)
    expected = x.sum(axis=0, keepdims=True)  # Sum along dim 0, keep all dimensions
    assert jnp.allclose(result, expected)


def test_empty_batch():
    """Test behavior with empty inputs."""
    # Single node, single feature
    x = jnp.array([[1.0]])
    batch = jnp.array([0])

    out = global_add_pool(x, batch)
    assert out.shape == (1, 1)
    assert jnp.allclose(out, x)

    out = global_mean_pool(x, batch)
    assert out.shape == (1, 1)
    assert jnp.allclose(out, x)

    out = global_max_pool(x, batch)
    assert out.shape == (1, 1)
    assert jnp.allclose(out, x)


def test_single_graph():
    """Test pooling operations on single graph."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Without batch vector (single graph)
    out_add = global_add_pool(x, None)
    out_mean = global_mean_pool(x, None)
    out_max = global_max_pool(x, None)

    assert out_add.shape == (1, 2)
    assert out_mean.shape == (1, 2)
    assert out_max.shape == (1, 2)

    assert jnp.allclose(out_add[0], jnp.array([9.0, 12.0]))  # Sum
    assert jnp.allclose(out_mean[0], jnp.array([3.0, 4.0]))  # Mean
    assert jnp.allclose(out_max[0], jnp.array([5.0, 6.0]))  # Max


def test_batch_size_parameter():
    """Test using the size parameter to avoid recomputing batch size."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    batch = jnp.array([0, 0, 1, 1])
    batch_size = 2

    # Test that providing size parameter works
    out1 = global_add_pool(x, batch, size=batch_size)
    out2 = global_add_pool(x, batch)  # Without size parameter

    assert out1.shape == (2, 2)
    assert out2.shape == (2, 2)
    assert jnp.allclose(out1, out2)


def test_large_batch():
    """Test pooling with larger batch sizes."""
    batch_size = 10
    nodes_per_graph = 5
    features = 8

    total_nodes = batch_size * nodes_per_graph
    x = jnp.arange(total_nodes * features, dtype=jnp.float32).reshape(total_nodes, features)
    batch = jnp.repeat(jnp.arange(batch_size), nodes_per_graph)

    # Test add pooling
    out_add = global_add_pool(x, batch)
    assert out_add.shape == (batch_size, features)

    # Test mean pooling
    out_mean = global_mean_pool(x, batch)
    assert out_mean.shape == (batch_size, features)

    # Test max pooling
    out_max = global_max_pool(x, batch)
    assert out_max.shape == (batch_size, features)

    # Verify correctness for first graph
    expected_add = x[:nodes_per_graph].sum(axis=0)
    expected_mean = x[:nodes_per_graph].mean(axis=0)
    expected_max = x[:nodes_per_graph].max(axis=0)

    assert jnp.allclose(out_add[0], expected_add)
    assert jnp.allclose(out_mean[0], expected_mean)
    assert jnp.allclose(out_max[0], expected_max)


def test_different_graph_sizes():
    """Test pooling with graphs of different sizes."""
    # Graph 0: 2 nodes, Graph 1: 3 nodes, Graph 2: 1 node
    x = jnp.array(
        [
            [1.0, 2.0],  # Graph 0
            [3.0, 4.0],
            [5.0, 6.0],  # Graph 1
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )  # Graph 2

    batch = jnp.array([0, 0, 1, 1, 1, 2])

    out = global_mean_pool(x, batch)
    assert out.shape == (3, 2)

    # Verify means
    assert jnp.allclose(out[0], jnp.array([2.0, 3.0]))  # (1+3)/2, (2+4)/2
    assert jnp.allclose(out[1], jnp.array([7.0, 8.0]))  # (5+7+9)/3, (6+8+10)/3
    assert jnp.allclose(out[2], jnp.array([11.0, 12.0]))  # Single node


# TODO: Additional pooling functions available in JraphX but not in PyG:
# - global_min_pool
# - global_softmax_pool
# - global_sort_pool
# - batch_histogram


def test_jraphx_extensions():
    """Test JraphX-specific pooling extensions."""
    from jraphx.nn.pool.glob import global_min_pool, global_softmax_pool

    x = jnp.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]])

    # Test min pooling
    out_min = global_min_pool(x, None)
    assert out_min.shape == (1, 2)
    assert jnp.allclose(out_min[0], jnp.array([1.0, 2.0]))  # Min values

    # Test softmax pooling (differentiable attention-based pooling)
    out_softmax = global_softmax_pool(x, None)
    assert out_softmax.shape == (1, 2)
    # Should be a weighted average, exact values depend on softmax weights


if __name__ == "__main__":
    # Run basic tests
    test_global_pool()
    test_permuted_global_pool()
    test_dense_global_pool()
    print("Global pooling tests passed!")
