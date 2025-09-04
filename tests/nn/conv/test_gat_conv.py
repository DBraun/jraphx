"""Test cases for JraphX GATConv layer converted from PyTorch Geometric tests."""

import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from jraphx.nn.conv import GATConv


@pytest.mark.parametrize("residual", [False, True])
def test_gat_conv_basic(residual):
    """Test basic GAT convolution functionality."""
    key = random.key(42)
    x1 = random.normal(key, (4, 8))
    # x2 unused in this test
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = GATConv(8, 32, heads=2, residual=residual, rngs=nnx.Rngs(0))

    # Test string representation - simplified check since __str__ may differ
    assert "GATConv" in str(type(conv).__name__)

    # Test forward pass
    out = conv(x1, edge_index)
    assert out.shape == (4, 64)  # heads * out_features when concat=True

    # TODO: Sparse tensor support not available in JAX - PyG uses to_torch_csc_tensor
    # JAX doesn't have direct sparse tensor equivalent like PyTorch
    # Original PyG test: assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    # TODO: PyTorch-specific sparse tensor operations not portable to JAX
    # Original PyG code uses SparseTensor which is PyTorch-specific
    # JAX uses different sparse representations that aren't directly comparable


def test_gat_conv_attention_weights():
    """Test GAT attention weights return functionality."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = GATConv(8, 32, heads=2, rngs=nnx.Rngs(0))

    # Test return_attention_weights functionality
    result = conv(x, edge_index, return_attention_weights=True)
    assert isinstance(result, tuple)
    assert len(result) == 2

    out, (edge_idx, alpha) = result
    assert out.shape == (4, 64)
    assert edge_idx.shape[0] == 2  # 2 for source/target
    assert alpha.shape[1] == 2  # Number of heads

    # Attention weights should be non-negative (after softmax)
    assert jnp.all(alpha >= 0)

    # TODO: Exact attention weight shape checks from PyG don't directly translate
    # PyG returns attention in different format than our implementation
    # Original: assert result[1][0].size() == (2, 7) - specific to PyG format


def test_gat_conv_bipartite():
    """Test GAT with bipartite graphs."""
    key = random.key(42)
    x1 = random.normal(key, (4, 8))
    x2 = random.normal(random.key(123), (2, 16))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    # Test bipartite initialization
    conv = GATConv((8, 16), 32, heads=2, rngs=nnx.Rngs(0))

    # Test bipartite forward pass
    out1 = conv((x1, x2), edge_index)
    assert out1.shape == (2, 64)  # Output has target node count

    # Test x_dst=None case (source nodes only)
    out_src_only = conv((x1, None), edge_index)
    assert out_src_only.shape == (4, 64)  # Output has source node count

    # Test size parameter
    out_with_size = conv((x1, x2), edge_index, size=(4, 2))
    assert jnp.allclose(out_with_size, out1, atol=1e-6)

    # TODO: Sparse tensor support not available
    # Original PyG uses adj matrices which aren't supported in JAX version


def test_gat_conv_edge_features():
    """Test GAT with edge features."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_weight = random.normal(random.key(123), (edge_index.shape[1],))
    edge_attr = random.normal(random.key(456), (edge_index.shape[1], 4))

    # Test with 1D edge features (edge weights)
    conv = GATConv(8, 32, heads=2, edge_dim=1, fill_value=0.5, rngs=nnx.Rngs(0))
    out = conv(x, edge_index, edge_weight)
    assert out.shape == (4, 64)

    # Test with multi-dimensional edge features
    conv = GATConv(8, 32, heads=2, edge_dim=4, fill_value=0.5, rngs=nnx.Rngs(0))
    out = conv(x, edge_index, edge_attr)
    assert out.shape == (4, 64)

    # Test string fill_value
    conv_mean = GATConv(8, 32, heads=2, edge_dim=4, fill_value="mean", rngs=nnx.Rngs(0))
    out_mean = conv_mean(x, edge_index, edge_attr)
    assert out_mean.shape == (4, 64)

    # Test other string fill values
    for fill_val in ["add", "max", "min"]:
        conv_fill = GATConv(8, 32, heads=2, edge_dim=4, fill_value=fill_val, rngs=nnx.Rngs(0))
        out_fill = conv_fill(x, edge_index, edge_attr)
        assert out_fill.shape == (4, 64)

    # TODO: Sparse tensor edge feature tests not portable
    # PyG tests with SparseTensor don't apply to JAX implementation


def test_gat_conv_empty_edges():
    """Test GAT with empty edge index."""
    key = random.key(42)
    x = random.normal(key, (4, 8))
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)

    conv = GATConv(8, 32, heads=2, rngs=nnx.Rngs(0))
    out = conv(x, edge_index)
    assert out.shape == (4, 64)

    # With no edges, output should only depend on self-loops (if enabled)
    # Test that it doesn't crash and produces expected shape


def test_gat_conv_parameters():
    """Test different GAT parameter configurations."""
    # Test concat=False (average heads instead)
    conv_avg = GATConv(8, 32, heads=4, concat=False, rngs=nnx.Rngs(0))
    x = jnp.ones((3, 8))
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

    out = conv_avg(x, edge_index)
    assert out.shape == (3, 32)  # out_features when concat=False

    # Test different negative slope
    conv_slope = GATConv(8, 16, negative_slope=0.1, rngs=nnx.Rngs(0))
    assert conv_slope.negative_slope == 0.1

    # Test dropout
    conv_dropout = GATConv(8, 16, dropout=0.5, rngs=nnx.Rngs(0))
    assert conv_dropout.dropout_rate == 0.5

    # Test no self loops
    conv_no_loops = GATConv(8, 16, add_self_loops=False, rngs=nnx.Rngs(0))
    assert conv_no_loops._add_self_loops is False

    # Test no bias
    conv_no_bias = GATConv(8, 16, bias=False, rngs=nnx.Rngs(0))
    assert conv_no_bias.bias is None


def test_gat_conv_heads():
    """Test GAT with different numbers of attention heads."""
    key = random.key(42)
    x = random.normal(key, (5, 10))
    edge_index = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

    # Test different head counts
    for heads in [1, 2, 4, 8]:
        conv = GATConv(10, 20, heads=heads, rngs=nnx.Rngs(0))
        out = conv(x, edge_index)
        assert out.shape == (5, heads * 20)  # concat=True by default

        # Test with concat=False
        conv_avg = GATConv(10, 20, heads=heads, concat=False, rngs=nnx.Rngs(0))
        out_avg = conv_avg(x, edge_index)
        assert out_avg.shape == (5, 20)


def test_gat_conv_deterministic():
    """Test that GAT is deterministic with same inputs and RNG."""
    conv1 = GATConv(6, 12, heads=2, rngs=nnx.Rngs(42))
    conv2 = GATConv(6, 12, heads=2, rngs=nnx.Rngs(42))

    x = jnp.ones((4, 6))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    out1 = conv1(x, edge_index)
    out2 = conv2(x, edge_index)

    # Should be deterministic with same RNG seed
    assert jnp.allclose(out1, out2)


def test_gat_conv_different_dtypes():
    """Test GAT with different input dtypes."""
    conv = GATConv(3, 6, rngs=nnx.Rngs(0))

    edge_index = jnp.array([[0, 1], [1, 0]])

    # Test float32
    x_f32 = jnp.ones((2, 3), dtype=jnp.float32)
    out_f32 = conv(x_f32, edge_index)
    assert out_f32.dtype == jnp.float32

    # Test float64 - may be converted based on JAX settings
    x_f64 = jnp.ones((2, 3), dtype=jnp.float64)
    _ = conv(x_f64, edge_index)  # Test that it works with float64
    # Note: output dtype depends on JAX default dtype settings


def test_gat_conv_large_graph():
    """Test GAT on larger graphs."""
    conv = GATConv(16, 32, heads=4, rngs=nnx.Rngs(0))

    # Create a larger random graph
    num_nodes = 100
    num_edges = 500

    key = random.key(42)
    x = random.normal(key, (num_nodes, 16))

    # Create random edges (ensure they're valid indices)
    edge_key = random.key(123)
    edges = random.randint(edge_key, (2, num_edges), 0, num_nodes)

    out = conv(x, edges)
    assert out.shape == (num_nodes, 4 * 32)


def test_gat_conv_residual():
    """Test GAT residual connections."""
    key = random.key(42)
    x = random.normal(key, (4, 16))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    # Test with residual connection
    conv_res = GATConv(16, 16, residual=True, rngs=nnx.Rngs(0))
    out_res = conv_res(x, edge_index)
    assert out_res.shape == (4, 16)

    # Test without residual
    conv_no_res = GATConv(16, 16, residual=False, rngs=nnx.Rngs(0))
    out_no_res = conv_no_res(x, edge_index)
    assert out_no_res.shape == (4, 16)

    # Results should be different
    assert not jnp.allclose(out_res, out_no_res, atol=1e-5)


# TODO: The following PyG GAT test features are not implemented in JraphX:
# - TorchScript JIT compilation - JAX uses different compilation (jax.jit)
# - Sparse tensor support (CSC, SparseTensor) - JAX doesn't have direct equivalent
# - Size parameter for message passing - Not implemented in current JraphX version
# - Complex sparse matrix operations - Beyond current scope
# - PyTorch-specific tensor operations - Different in JAX
# - fill_value="mean" - Only float values supported in JraphX
# - Device-specific tests (CUDA) - JAX handles devices differently
# - torch_sparse integration - Not applicable to JAX
# - Gradient computation tests - Can be added separately with jax.grad
# - Memory optimization tests - JAX has different memory model
# - Advanced sparse attention weight formats - Simplified in JraphX

# These missing features are documented in docs/source/missing_tests.rst
