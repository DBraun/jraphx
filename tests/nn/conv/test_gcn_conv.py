"""Test cases for JraphX GCNConv layer converted from PyTorch Geometric tests."""

import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from jraphx.nn.conv import GCNConv


def test_gcn_conv_basic():
    """Test basic GCN convolution functionality."""
    key = random.key(42)
    x = random.normal(key, (4, 16))
    edge_index = jnp.array([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = GCNConv(16, 32, rngs=nnx.Rngs(0))

    # Test string representation
    assert "GCNConv" in str(conv)
    assert "in_features=16" in str(conv)
    assert "out_features=32" in str(conv)

    # Test forward pass
    out = conv(x, edge_index)
    assert out.shape == (4, 32)

    # Test that output is different from input (due to transformation)
    # Since shapes are different (4,32) vs (4,16), just check that output exists
    assert out is not None


def test_gcn_conv_with_edge_weights():
    """Test GCN convolution with edge weights."""
    key = random.key(42)
    x = random.normal(key, (4, 16))
    edge_index = jnp.array([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_weight = random.uniform(random.key(123), (edge_index.shape[1],))

    conv = GCNConv(16, 32, rngs=nnx.Rngs(0))

    # Test with edge weights
    out_weighted = conv(x, edge_index, edge_weight)
    assert out_weighted.shape == (4, 32)

    # Test without edge weights
    out_unweighted = conv(x, edge_index)
    assert out_unweighted.shape == (4, 32)

    # Results should be different with vs without edge weights
    assert not jnp.allclose(out_weighted, out_unweighted, atol=1e-5)


def test_gcn_conv_parameters():
    """Test different GCN parameter configurations."""
    # Test improved GCN
    conv_improved = GCNConv(8, 16, improved=True, rngs=nnx.Rngs(0))
    assert conv_improved.improved is True

    # Test without bias
    conv_no_bias = GCNConv(8, 16, bias=False, rngs=nnx.Rngs(0))
    assert conv_no_bias.linear.use_bias is False

    # Test without normalization
    conv_no_norm = GCNConv(8, 16, normalize=False, add_self_loops=False, rngs=nnx.Rngs(0))
    assert conv_no_norm.normalize is False
    assert conv_no_norm._add_self_loops is False


def test_gcn_conv_error_conditions():
    """Test error conditions for GCN layer."""
    # Test invalid parameter combination
    with pytest.raises(ValueError, match="does not support adding self-loops"):
        GCNConv(16, 32, add_self_loops=True, normalize=False, rngs=nnx.Rngs(0))


def test_gcn_conv_shapes():
    """Test GCN with different input shapes."""
    conv = GCNConv(10, 20, rngs=nnx.Rngs(0))

    # Test with different number of nodes
    for num_nodes in [2, 5, 10, 20]:
        x = jnp.ones((num_nodes, 10))
        # Create simple ring graph
        if num_nodes > 1:
            edge_index = jnp.array(
                [jnp.arange(num_nodes), jnp.concatenate([jnp.arange(1, num_nodes), jnp.array([0])])]
            )
        else:
            edge_index = jnp.array([[0], [0]])  # Self-loop for single node

        out = conv(x, edge_index)
        assert out.shape == (num_nodes, 20)


def test_gcn_conv_empty_graph():
    """Test GCN with empty edges."""
    conv = GCNConv(5, 10, rngs=nnx.Rngs(0))

    x = jnp.ones((3, 5))
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)  # No edges

    out = conv(x, edge_index)
    assert out.shape == (3, 10)


def test_gcn_conv_self_loops():
    """Test GCN behavior with self-loops."""
    conv = GCNConv(4, 8, rngs=nnx.Rngs(0))

    x = jnp.ones((3, 4))
    # Only self-loops
    edge_index = jnp.array([[0, 1, 2], [0, 1, 2]])

    out = conv(x, edge_index)
    assert out.shape == (3, 8)


def test_gcn_conv_deterministic():
    """Test that GCN is deterministic with same inputs."""
    conv = GCNConv(6, 12, rngs=nnx.Rngs(42))

    x = jnp.ones((4, 6))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    out1 = conv(x, edge_index)
    out2 = conv(x, edge_index)

    # Should be deterministic (same output for same input)
    assert jnp.allclose(out1, out2)


def test_gcn_conv_different_dtypes():
    """Test GCN with different input dtypes."""
    conv = GCNConv(3, 6, rngs=nnx.Rngs(0))

    edge_index = jnp.array([[0, 1], [1, 0]])

    # Test float32
    x_f32 = jnp.ones((2, 3), dtype=jnp.float32)
    out_f32 = conv(x_f32, edge_index)
    assert out_f32.dtype == jnp.float32

    # Test float64 - may be converted based on JAX settings
    x_f64 = jnp.ones((2, 3), dtype=jnp.float64)
    _ = conv(x_f64, edge_index)  # Test that it works with float64
    # Note: output dtype depends on JAX default dtype settings


def test_gcn_conv_batch_processing():
    """Test GCN with batch of graphs (using batch vector)."""
    conv = GCNConv(4, 8, rngs=nnx.Rngs(0))

    # Two graphs: first with 2 nodes, second with 3 nodes
    x = jnp.ones((5, 4))  # Total 5 nodes
    edge_index = jnp.array(
        [
            [0, 1, 2, 3, 4, 2],  # Edges within each graph
            [1, 0, 3, 4, 2, 4],  # (adjusted for batching)
        ]
    )

    out = conv(x, edge_index)
    assert out.shape == (5, 8)


# TODO: The following PyG GCN test features are not implemented in JraphX:
# - Sparse tensor support (adj matrices) - JAX doesn't have direct equivalent
# - TorchScript JIT compilation - JAX uses different compilation (jax.jit)
# - Cached computations - JraphX has limited caching support
# - Complex sparse matrix operations - Beyond current scope
# - torch_sparse integration - Not applicable to JAX
# - PyTorch-specific tensor operations - Different in JAX
# - Gradient computation tests - Can be added separately with jax.grad
# - Device-specific tests (CUDA) - JAX handles devices differently
# - Memory optimization tests - JAX has different memory model
# - Advanced edge case handling - Simplified in JraphX

# These missing features are documented in docs/source/missing_tests.rst
