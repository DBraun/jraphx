"""Test cases for JraphX SAGEConv layer converted from PyTorch Geometric tests."""

import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from jraphx.nn.conv import SAGEConv


@pytest.mark.parametrize("root_weight", [False, True])
@pytest.mark.parametrize("aggr", ["mean", "max"])
def test_sage_conv(root_weight, aggr):
    """Test basic SAGE convolution functionality."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = SAGEConv(8, 32, root_weight=root_weight, aggr=aggr, rngs=nnx.Rngs(0))
    assert f"aggr={aggr}" in str(conv) or "SAGEConv" in str(conv)

    out = conv(x, edge_index)
    assert out.shape == (4, 32)

    # Test that output is not all zeros
    assert not jnp.allclose(out, jnp.zeros_like(out))

    # Test deterministic behavior
    out2 = conv(x, edge_index)
    assert jnp.allclose(out, out2)

    # TODO: JIT compilation testing - JAX uses jax.jit instead of torch.jit.script
    # TODO: Sparse tensor support - JAX doesn't have direct equivalent to SparseTensor
    # TODO: Size parameter usage - Need to verify if this is properly supported


def test_sage_conv_bipartite():
    """Test SAGE convolution with bipartite message passing."""

    # Test basic bipartite initialization
    conv = SAGEConv((8, 16), 32, rngs=nnx.Rngs(0))
    assert "SAGEConv" in str(conv)
    assert conv.in_features == (8, 16)

    # Test full bipartite message passing
    x1 = jnp.ones((4, 8))  # Source nodes
    x2 = jnp.ones((2, 16))  # Target nodes
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    out1 = conv((x1, x2), edge_index, size=(4, 2))
    assert out1.shape == (2, 32)  # Should have target node count


def test_sage_conv_lazy():
    """Test lazy initialization behavior."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    # TODO: Lazy initialization not supported in current JraphX implementation
    # PyG supports -1 for lazy initialization, but JraphX requires explicit dimensions
    # This would need to be implemented in JraphX to match PyG behavior

    # Test normal initialization works
    conv = SAGEConv(8, 32, rngs=nnx.Rngs(0))
    out = conv(x, edge_index)
    assert out.shape == (4, 32)


def test_lstm_aggr_sage_conv():
    """Test LSTM aggregation in SAGE convolution."""
    # TODO: LSTM aggregation not implemented in JraphX yet
    # This would raise NotImplementedError in current implementation
    with pytest.raises(NotImplementedError, match="LSTM aggregation is not yet implemented"):
        SAGEConv(8, 32, aggr="lstm", rngs=nnx.Rngs(0))

    # TODO: Edge sorting validation - PyG checks if edges are sorted for LSTM
    # This would need to be implemented when LSTM support is added


def test_mlp_sage_conv():
    """Test SAGE convolution with MLP aggregation."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    # TODO: MLPAggregation not implemented in JraphX
    # PyG has MLPAggregation class that can be passed as aggr parameter
    # JraphX currently only supports basic aggregation types: mean, max, sum

    # Test that basic aggregation still works
    conv = SAGEConv(8, 32, aggr="mean", rngs=nnx.Rngs(0))
    out = conv(x, edge_index)
    assert out.shape == (4, 32)


def test_multi_aggr_sage_conv():
    """Test SAGE convolution with multiple aggregation methods."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    # TODO: Multiple aggregation not supported in JraphX
    # PyG allows aggr=["mean", "max", "sum", "softmax"] with various modes
    # JraphX currently supports only single aggregation method

    # Test single aggregation methods that are supported
    for aggr in ["mean", "max"]:
        conv = SAGEConv(8, 32, aggr=aggr, rngs=nnx.Rngs(0))
        out = conv(x, edge_index)
        assert out.shape == (4, 32)


def test_sage_conv_normalization():
    """Test SAGE convolution with L2 normalization."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv_norm = SAGEConv(8, 32, normalize=True, rngs=nnx.Rngs(0))
    conv_no_norm = SAGEConv(8, 32, normalize=False, rngs=nnx.Rngs(1))

    out_norm = conv_norm(x, edge_index)
    out_no_norm = conv_no_norm(x, edge_index)

    # Check that normalized output has unit norm (approximately)
    norms = jnp.linalg.norm(out_norm, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)

    # Non-normalized output should have different norms
    norms_no_norm = jnp.linalg.norm(out_no_norm, axis=-1)
    assert not jnp.allclose(norms_no_norm, 1.0, atol=1e-2)


def test_sage_conv_gcn_style():
    """Test SAGE convolution with GCN-style aggregation."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv_gcn = SAGEConv(8, 32, aggr="gcn", rngs=nnx.Rngs(0))
    conv_mean = SAGEConv(8, 32, aggr="mean", rngs=nnx.Rngs(0))

    out_gcn = conv_gcn(x, edge_index)
    out_mean = conv_mean(x, edge_index)

    assert out_gcn.shape == (4, 32)
    assert out_mean.shape == (4, 32)

    # GCN and mean aggregation should produce different results
    assert not jnp.allclose(out_gcn, out_mean, atol=1e-3)


def test_sage_conv_parameters():
    """Test different SAGE parameter configurations."""
    # Test without bias
    conv_no_bias = SAGEConv(8, 16, bias=False, rngs=nnx.Rngs(0))
    assert conv_no_bias.lin.use_bias is False

    # Test without root weight
    conv_no_root = SAGEConv(8, 16, root_weight=False, rngs=nnx.Rngs(0))
    assert conv_no_root.root_weight is False
    assert conv_no_root.lin_r is None


def test_sage_conv_shapes():
    """Test SAGE with different input shapes."""
    conv = SAGEConv(10, 20, rngs=nnx.Rngs(0))

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


def test_sage_conv_empty_graph():
    """Test SAGE with empty edges."""
    conv = SAGEConv(5, 10, rngs=nnx.Rngs(0))

    x = jnp.ones((3, 5))
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)  # No edges

    out = conv(x, edge_index)
    assert out.shape == (3, 10)


def test_sage_conv_different_dtypes():
    """Test SAGE with different input dtypes."""
    conv = SAGEConv(3, 6, rngs=nnx.Rngs(0))

    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    # Test float32
    x_f32 = jnp.ones((2, 3), dtype=jnp.float32)
    out_f32 = conv(x_f32, edge_index)
    assert out_f32.dtype == jnp.float32

    # Test float64 - may be converted based on JAX settings
    x_f64 = jnp.ones((2, 3), dtype=jnp.float64)
    _ = conv(x_f64, edge_index)  # Test that it works with float64
    # Note: output dtype depends on JAX default dtype settings


def test_sage_conv_deterministic():
    """Test that SAGE is deterministic with same inputs."""
    conv = SAGEConv(6, 12, rngs=nnx.Rngs(42))

    x = jnp.ones((4, 6))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    out1 = conv(x, edge_index)
    out2 = conv(x, edge_index)

    # Should be deterministic (same output for same input)
    assert jnp.allclose(out1, out2)


# TODO: The following PyG SAGE test features are not implemented in JraphX:
# - TorchScript JIT compilation - JAX uses jax.jit with different syntax
# - Sparse tensor support (SparseTensor) - JAX doesn't have direct equivalent
# - Multiple aggregation methods in single layer - Not supported
# - MLPAggregation - Not implemented
# - LSTM aggregation - Not implemented
# - Lazy initialization with -1 dimensions - Not supported
# - Advanced aggregation kwargs and modes - Simplified in JraphX
# - PyTorch-specific tensor operations - Different in JAX
# - Dynamic compilation testing - JAX has different compilation model
# - Complex bipartite graph handling - Partially supported, shape mismatch issues
# - Edge sorting validation for LSTM - Not applicable without LSTM
# - Custom aggregation functions - Limited support
# - Device-specific testing - JAX handles devices differently
# - Memory optimization patterns - JAX has different memory model

# Key differences in JraphX implementation:
# - Uses flax.nnx.Linear instead of torch.nn.Linear
# - Normalization uses jnp.linalg.norm instead of torch.nn.functional.normalize
# - Random initialization uses nnx.Rngs instead of torch default initialization
# - Bipartite message passing has different shape handling requirements
# - Aggregation operations use custom JAX scatter functions
# - Parameter access patterns differ (e.g., conv.lin.use_bias vs conv.lin.bias)

# These missing features are documented in docs/source/missing_features.rst
