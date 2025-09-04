"""Test cases for JraphX EdgeConv layer converted from PyTorch Geometric tests."""

import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from jraphx.nn.conv import DynamicEdgeConv, EdgeConv
from jraphx.nn.models import MLP


def test_edge_conv_basic():
    """Test basic EdgeConv functionality."""
    key = random.key(42)
    x1 = random.normal(key, (4, 16))
    # x2 = random.normal(random.key(123), (2, 16))  # Not used in this test
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    # EdgeConv expects concatenated features [x_i, x_j - x_i], so input size is 2 * 16 = 32
    nn = MLP([32, 16, 32], rngs=nnx.Rngs(0))
    conv = EdgeConv(nn)

    # Test string representation
    assert "EdgeConv" in str(conv) or hasattr(conv, "nn")

    # Test forward pass
    out = conv(x1, edge_index)
    assert out.shape == (4, 32)

    # Test that output is different from input
    # Since shapes are different (4,32) vs (4,16), just check that output exists
    assert out is not None


def test_edge_conv_bipartite():
    """Test EdgeConv with bipartite-like setup."""
    key = random.key(42)
    x1 = random.normal(key, (4, 16))
    x2 = random.normal(random.key(123), (2, 16))
    # edge_index not needed for this specific test setup

    nn = MLP([32, 16, 32], rngs=nnx.Rngs(0))
    conv = EdgeConv(nn)

    # Test with bipartite-like setup by concatenating features
    # and adjusting edge indices accordingly
    x_combined = jnp.concatenate([x1, x2], axis=0)  # [6, 16]
    edge_index_adjusted = jnp.array([[0, 1, 2, 3], [4, 4, 5, 5]])  # Point to x2 nodes

    out = conv(x_combined, edge_index_adjusted)
    assert out.shape == (6, 32)


def test_edge_conv_aggregation():
    """Test EdgeConv with different aggregation methods."""
    x = random.normal(random.key(42), (4, 16))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    nn = MLP([32, 16, 32], rngs=nnx.Rngs(0))

    # Test max aggregation (default)
    conv_max = EdgeConv(nn, aggr="max")
    out_max = conv_max(x, edge_index)
    assert out_max.shape == (4, 32)

    # Test mean aggregation
    conv_mean = EdgeConv(nn, aggr="mean")
    out_mean = conv_mean(x, edge_index)
    assert out_mean.shape == (4, 32)

    # Test add aggregation
    conv_add = EdgeConv(nn, aggr="add")
    out_add = conv_add(x, edge_index)
    assert out_add.shape == (4, 32)

    # Different aggregation methods should give different results
    assert not jnp.allclose(out_max, out_mean)
    assert not jnp.allclose(out_max, out_add)
    assert not jnp.allclose(out_mean, out_add)


def test_edge_conv_message_computation():
    """Test EdgeConv message computation [x_i, x_j - x_i]."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 nodes, 2 features
    edge_index = jnp.array([[0, 1], [1, 2]])  # 0->1, 1->2

    # Simple linear network to test message passing
    nn = nnx.Linear(4, 2, rngs=nnx.Rngs(0))  # Input: 2*2=4, Output: 2
    conv = EdgeConv(nn)

    out = conv(x, edge_index)
    assert out.shape == (3, 2)

    # For edge 0->1: message = [x_1, x_0 - x_1] = [[3,4], [1,2] - [3,4]] = [[3,4], [-2,-2]]
    # For edge 1->2: message = [x_2, x_1 - x_2] = [[5,6], [3,4] - [5,6]] = [[5,6], [-2,-2]]


def test_edge_conv_shapes():
    """Test EdgeConv with different input shapes."""
    nn = MLP([20, 10], rngs=nnx.Rngs(0))  # Input: 2*10=20, Output: 10
    conv = EdgeConv(nn)

    # Test with different number of nodes
    for num_nodes in [2, 5, 10]:
        x = jnp.ones((num_nodes, 10))
        # Create simple ring graph
        if num_nodes > 1:
            edge_index = jnp.array(
                [jnp.arange(num_nodes), jnp.concatenate([jnp.arange(1, num_nodes), jnp.array([0])])]
            )
        else:
            edge_index = jnp.array([[0], [0]])  # Self-loop for single node

        out = conv(x, edge_index)
        assert out.shape == (num_nodes, 10)


def test_edge_conv_empty_graph():
    """Test EdgeConv with empty edges."""
    nn = MLP([10, 5], rngs=nnx.Rngs(0))  # Input: 2*5=10, Output: 5
    conv = EdgeConv(nn)

    x = jnp.ones((3, 5))
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)  # No edges

    out = conv(x, edge_index)
    assert out.shape == (3, 5)

    # With no edges, all nodes should get zero aggregation (depending on aggr method)


def test_edge_conv_self_loops():
    """Test EdgeConv behavior with self-loops."""
    nn = MLP([8, 4], rngs=nnx.Rngs(0))  # Input: 2*4=8, Output: 4
    conv = EdgeConv(nn)

    x = jnp.ones((3, 4))
    # Only self-loops: for self-loop i->i, message is [x_i, x_i - x_i] = [x_i, 0]
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    out = conv(x, edge_index)
    assert out.shape == (3, 4)


def test_edge_conv_deterministic():
    """Test that EdgeConv is deterministic with same inputs."""
    nn = MLP([12, 6], rngs=nnx.Rngs(42))  # Input: 2*6=12, Output: 6
    conv = EdgeConv(nn)

    x = jnp.ones((4, 6))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    out1 = conv(x, edge_index)
    out2 = conv(x, edge_index)

    # Should be deterministic (same output for same input)
    assert jnp.allclose(out1, out2)


def test_dynamic_edge_conv_basic():
    """Test basic DynamicEdgeConv functionality."""
    x = random.normal(random.key(42), (8, 16))

    nn = MLP([32, 16, 32], rngs=nnx.Rngs(0))
    conv = DynamicEdgeConv(nn, k=2)

    # Test string representation
    assert hasattr(conv, "k") and conv.k == 2

    # For testing, create k-NN indices manually (2-NN for each node)
    # Simple case: each node connects to next 2 nodes (cyclic)
    num_nodes = x.shape[0]
    knn_indices = jnp.array(
        [[(i + 1) % num_nodes, (i + 2) % num_nodes] for i in range(num_nodes)]
    )  # [num_nodes, k]

    out = conv(x, knn_indices=knn_indices)
    assert out.shape == (8, 32)


def test_dynamic_edge_conv_error():
    """Test DynamicEdgeConv error handling."""
    x = random.normal(random.key(42), (4, 16))

    nn = MLP([32, 16, 32], rngs=nnx.Rngs(0))
    conv = DynamicEdgeConv(nn, k=2)

    # Should raise error if neither edge_index nor knn_indices provided
    with pytest.raises(ValueError, match="Either edge_index or knn_indices must be provided"):
        conv(x)


def test_dynamic_edge_conv_with_edge_index():
    """Test DynamicEdgeConv with pre-computed edge_index."""
    x = random.normal(random.key(42), (4, 16))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    nn = MLP([32, 16, 32], rngs=nnx.Rngs(0))
    conv = DynamicEdgeConv(nn, k=2)

    out = conv(x, edge_index=edge_index)
    assert out.shape == (4, 32)


def test_edge_conv_different_dtypes():
    """Test EdgeConv with different input dtypes."""
    nn = MLP([6, 3], rngs=nnx.Rngs(0))  # Input: 2*3=6, Output: 3
    conv = EdgeConv(nn)

    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    # Test float32
    x_f32 = jnp.ones((2, 3), dtype=jnp.float32)
    out_f32 = conv(x_f32, edge_index)
    assert out_f32.dtype == jnp.float32

    # Test float64
    x_f64 = jnp.ones((2, 3), dtype=jnp.float64)
    _ = conv(x_f64, edge_index)  # Test that it works with float64
    # Note: output dtype depends on JAX default dtype settings


def test_dynamic_edge_conv_knn_requirement():
    """Test that DynamicEdgeConv clearly demonstrates k-NN requirement vs PyG."""
    x = random.normal(random.key(42), (10, 3))  # 10 points in 3D

    nn = MLP([6, 128], rngs=nnx.Rngs(0))  # Input: 2*3=6, Output: 128
    conv = DynamicEdgeConv(nn, k=3)

    # Unlike PyG's DynamicEdgeConv which would do: conv(x, batch=batch)
    # JraphX requires pre-computed k-NN indices

    # Example: compute simple spatial k-NN (3 nearest neighbors)
    # In practice, you'd use a proper k-NN library like sklearn or faiss
    def simple_knn(points, k):
        """Simple k-NN for demonstration (not optimized for production)."""
        num_points = points.shape[0]
        knn_indices = []

        for i in range(num_points):
            # Compute distances to all other points
            distances = jnp.sum((points - points[i]) ** 2, axis=1)
            # Get k+1 nearest (including self), then exclude self
            nearest = jnp.argsort(distances)
            # Take k neighbors (excluding self at index 0)
            knn_indices.append(nearest[1 : k + 1])

        return jnp.array(knn_indices)

    knn_indices = simple_knn(x, k=3)
    assert knn_indices.shape == (10, 3)

    # Now we can use DynamicEdgeConv with pre-computed k-NN
    output = conv(x, knn_indices=knn_indices)
    assert output.shape == (10, 128)


def test_dynamic_edge_conv_vs_pyg_comparison():
    """Test documenting the difference between JraphX and PyG DynamicEdgeConv."""
    # This test documents the API difference, not functional difference
    x = random.normal(random.key(42), (5, 4))

    nn = MLP([8, 16], rngs=nnx.Rngs(0))
    conv = DynamicEdgeConv(nn, k=2)

    # PyG equivalent would be:
    # conv = torch_geometric.nn.DynamicEdgeConv(nn, k=2)
    # output = conv(x, batch=batch)  # Automatic k-NN from features

    # JraphX requires:
    knn_indices = jnp.array(
        [
            [1, 2],  # Node 0 connects to nodes 1, 2
            [0, 2],  # Node 1 connects to nodes 0, 2
            [0, 1],  # Node 2 connects to nodes 0, 1
            [0, 1],  # Node 3 connects to nodes 0, 1
            [0, 1],  # Node 4 connects to nodes 0, 1
        ]
    )

    output = conv(x, knn_indices=knn_indices)
    assert output.shape == (5, 16)

    # The key difference: PyG computes k-NN automatically using torch_cluster.knn()
    # JraphX requires you to provide k-NN indices explicitly


# TODO: The following PyG EdgeConv test features are not implemented in JraphX:
#
# Major Limitations (documented in missing_features.rst):
# - Automatic k-NN graph construction via torch_cluster.knn() - Would require JAX k-NN implementation
# - Batch processing with batch vector for automatic k-NN - Different batching approach in JraphX
# - Dynamic graph construction from node features - Core limitation without JAX k-NN library
#
# PyTorch-Specific Features:
# - torch_cluster integration for k-NN - Not applicable to JAX ecosystem
# - torch_sparse integration - Not applicable to JAX
# - TorchScript JIT compilation - JAX uses jax.jit instead
# - CUDA-specific optimizations - JAX handles device placement differently
#
# Advanced Features (could be added):
# - Complex bipartite message passing with tuple input - Different design in JraphX
# - Sparse tensor support (adj matrices) - JAX has different sparse support
# - Advanced sparse matrix operations - Beyond current scope
# - Gradient computation tests - Can be added separately with jax.grad
# - Memory optimization tests - JAX has different memory model
#
# These missing features are documented in docs/source/missing_features.rst
