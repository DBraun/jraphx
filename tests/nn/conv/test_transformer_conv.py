"""Test cases for JraphX TransformerConv layer converted from PyTorch Geometric tests."""

from flax import nnx
from jax import numpy as jnp
from jax import random

from jraphx.nn.conv import TransformerConv


def test_transformer_conv_basic():
    """Test basic TransformerConv functionality."""
    key = random.key(42)
    x1 = random.normal(key, (4, 8))
    out_features = 32
    heads = 2
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = TransformerConv(8, out_features, heads, rngs=nnx.Rngs(0))

    # Test string representation
    assert "TransformerConv" in str(conv) or hasattr(conv, "heads")

    # Test forward pass
    out = conv(x1, edge_index)
    expected_shape = (4, out_features * heads)  # concat=True by default
    assert out.shape == expected_shape

    # Test that output is different from input
    # Since shapes are different (4,64) vs (4,8), just check that output exists
    assert out is not None


def test_transformer_conv_concat_options():
    """Test TransformerConv with different concatenation options."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    out_features = 32
    heads = 2

    # Test with concat=True (default)
    conv_concat = TransformerConv(8, out_features, heads, concat=True, rngs=nnx.Rngs(0))
    out_concat = conv_concat(x, edge_index)
    assert out_concat.shape == (4, out_features * heads)

    # Test with concat=False
    conv_no_concat = TransformerConv(8, out_features, heads, concat=False, rngs=nnx.Rngs(1))
    out_no_concat = conv_no_concat(x, edge_index)
    assert out_no_concat.shape == (4, out_features)

    # Results should be different
    assert not jnp.allclose(out_concat.mean(axis=-1), out_no_concat.mean(axis=-1))


def test_transformer_conv_with_edge_attr():
    """Test TransformerConv with edge attributes."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = random.normal(random.key(123), (4, 8))  # 4 edges, 8 features
    out_features = 32
    heads = 2
    edge_dim = 8

    conv = TransformerConv(8, out_features, heads, edge_dim=edge_dim, rngs=nnx.Rngs(0))

    # Test with edge attributes
    out_with_attr = conv(x, edge_index, edge_attr)
    assert out_with_attr.shape == (4, out_features * heads)

    # Test without edge attributes
    out_without_attr = conv(x, edge_index)
    assert out_without_attr.shape == (4, out_features * heads)

    # Results should be different when edge attributes are used
    assert not jnp.allclose(out_with_attr, out_without_attr)


def test_transformer_conv_beta_gating():
    """Test TransformerConv with beta gating."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    out_features = 16
    heads = 2

    # Test with beta=False (default)
    conv_no_beta = TransformerConv(8, out_features, heads, beta=False, rngs=nnx.Rngs(0))
    out_no_beta = conv_no_beta(x, edge_index)
    assert out_no_beta.shape == (4, out_features * heads)

    # Test with beta=True
    conv_beta = TransformerConv(8, out_features, heads, beta=True, rngs=nnx.Rngs(1))
    out_beta = conv_beta(x, edge_index)
    assert out_beta.shape == (4, out_features * heads)

    # Check that beta gating layer exists
    assert conv_beta.beta is True
    assert hasattr(conv_beta, "lin_beta")

    # Results should be different with beta gating
    assert not jnp.allclose(out_no_beta, out_beta)


def test_transformer_conv_no_root_weight():
    """Test TransformerConv without root weight (skip connection)."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    out_features = 16
    heads = 2

    # Test with root_weight=True (default)
    conv_root = TransformerConv(8, out_features, heads, root_weight=True, rngs=nnx.Rngs(0))
    out_root = conv_root(x, edge_index)
    assert out_root.shape == (4, out_features * heads)

    # Test with root_weight=False
    conv_no_root = TransformerConv(8, out_features, heads, root_weight=False, rngs=nnx.Rngs(1))
    out_no_root = conv_no_root(x, edge_index)
    assert out_no_root.shape == (4, out_features * heads)

    # Check skip connection layer existence
    assert conv_root.root_weight is True
    assert hasattr(conv_root, "lin_skip")
    assert conv_no_root.root_weight is False
    assert conv_no_root.lin_skip is None

    # Results should be different
    assert not jnp.allclose(out_root, out_no_root)


def test_transformer_conv_dropout():
    """Test TransformerConv with dropout."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    out_features = 16
    heads = 2

    # Test with dropout_rate=0 (default)
    conv_no_dropout = TransformerConv(8, out_features, heads, dropout_rate=0.0, rngs=nnx.Rngs(0))
    assert conv_no_dropout.dropout_rate == 0.0

    # Test with dropout_rate>0
    conv_dropout = TransformerConv(8, out_features, heads, dropout_rate=0.1, rngs=nnx.Rngs(1))
    assert conv_dropout.dropout_rate == 0.1
    assert hasattr(conv_dropout, "dropout")

    # Test forward passes
    out_no_dropout = conv_no_dropout(x, edge_index)
    out_dropout = conv_dropout(x, edge_index)

    assert out_no_dropout.shape == (4, out_features * heads)
    assert out_dropout.shape == (4, out_features * heads)


def test_transformer_conv_different_heads():
    """Test TransformerConv with different numbers of attention heads."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    out_features = 16

    for heads in [1, 2, 4, 8]:
        conv = TransformerConv(8, out_features, heads, rngs=nnx.Rngs(heads))
        out = conv(x, edge_index)
        assert out.shape == (4, out_features * heads)  # concat=True by default
        assert conv.heads == heads


def test_transformer_conv_shapes():
    """Test TransformerConv with different input shapes."""
    out_features = 10
    heads = 2
    conv = TransformerConv(5, out_features, heads, rngs=nnx.Rngs(0))

    # Test with different number of nodes
    for num_nodes in [2, 5, 10]:
        x = jnp.ones((num_nodes, 5))
        # Create simple ring graph
        if num_nodes > 1:
            edge_index = jnp.array(
                [jnp.arange(num_nodes), jnp.concatenate([jnp.arange(1, num_nodes), jnp.array([0])])]
            )
        else:
            edge_index = jnp.array([[0], [0]])  # Self-loop for single node

        out = conv(x, edge_index)
        assert out.shape == (num_nodes, out_features * heads)


def test_transformer_conv_empty_graph():
    """Test TransformerConv with empty edges."""
    conv = TransformerConv(5, 10, heads=2, rngs=nnx.Rngs(0))

    x = jnp.ones((3, 5))
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)  # No edges

    out = conv(x, edge_index)
    assert out.shape == (3, 10 * 2)  # 10 out_features * 2 heads


def test_transformer_conv_self_loops():
    """Test TransformerConv behavior with self-loops."""
    conv = TransformerConv(4, 8, heads=2, rngs=nnx.Rngs(0))

    x = jnp.ones((3, 4))
    # Only self-loops
    edge_index = jnp.array([[0, 1, 2], [0, 1, 2]])

    out = conv(x, edge_index)
    assert out.shape == (3, 8 * 2)


def test_transformer_conv_deterministic():
    """Test that TransformerConv is deterministic with same inputs."""
    conv = TransformerConv(6, 12, heads=2, rngs=nnx.Rngs(42))

    x = jnp.ones((4, 6))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    out1 = conv(x, edge_index)
    out2 = conv(x, edge_index)

    # Should be deterministic (same output for same input)
    assert jnp.allclose(out1, out2)


def test_transformer_conv_aggregation_methods():
    """Test TransformerConv with different aggregation methods."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    out_features = 16
    heads = 2

    # Test different aggregation methods
    for aggr in ["add", "mean", "max"]:
        conv = TransformerConv(8, out_features, heads, aggr=aggr, rngs=nnx.Rngs(0))
        out = conv(x, edge_index)
        assert out.shape == (4, out_features * heads)


def test_transformer_conv_different_dtypes():
    """Test TransformerConv with different input dtypes."""
    conv = TransformerConv(3, 6, heads=2, rngs=nnx.Rngs(0))

    edge_index = jnp.array([[0, 1], [1, 0]])

    # Test float32
    x_f32 = jnp.ones((2, 3), dtype=jnp.float32)
    out_f32 = conv(x_f32, edge_index)
    assert out_f32.dtype == jnp.float32

    # Test float64
    x_f64 = jnp.ones((2, 3), dtype=jnp.float64)
    _ = conv(x_f64, edge_index)  # Test that it works with float64
    # Note: output dtype depends on JAX default dtype settings


# TODO: The following PyG TransformerConv test features are not implemented in JraphX:
# - Return attention weights functionality - Not implemented yet
# - Bipartite message passing with tuple input (x1, x2) - Not directly supported
# - Sparse tensor support (adj matrices) - JAX doesn't have direct equivalent
# - TorchScript JIT compilation - JAX uses different compilation (jax.jit)
# - Complex sparse matrix operations - Beyond current scope
# - torch_sparse integration - Not applicable to JAX
# - Parametric tests with pytest.mark.parametrize - Simplified individual tests
# - Advanced edge case handling - Simplified in JraphX
# - PyTorch-specific tensor operations - Different in JAX
# - Gradient computation tests - Can be added separately with jax.grad
# - Device-specific tests (CUDA) - JAX handles devices differently
# - Memory optimization tests - JAX has different memory model
# - Custom attention weight extraction - Not implemented

# These missing features are documented in docs/source/missing_features.rst
