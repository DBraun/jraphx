"""Test cases for JraphX TransformerConv layer converted from PyTorch Geometric tests."""

import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from jraphx.nn.conv import TransformerConv
from jraphx.utils import scatter_add, scatter_softmax


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

    # Different edge attributes must produce different outputs
    out_other_attr = conv(x, edge_index, edge_attr * 2.0)
    assert not jnp.allclose(out_with_attr, out_other_attr)

    # A layer built with edge_dim requires edge features
    with pytest.raises(RuntimeError):
        conv(x, edge_index)


def test_transformer_conv_edge_attr_conditions_attention():
    """Edge features enter the keys, so they reshape the attention distribution."""
    heads, out_features, edge_dim = 2, 4, 3
    num_nodes = 5
    x = random.normal(random.key(1), (num_nodes, 6))
    edge_index = jnp.array([[0, 1, 2, 3, 4, 1], [1, 1, 1, 4, 4, 4]])
    edge_attr = random.normal(random.key(2), (edge_index.shape[1], edge_dim))

    conv = TransformerConv(6, out_features, heads, edge_dim=edge_dim, rngs=nnx.Rngs(0))
    out = conv(x, edge_index, edge_attr)

    # Reference implementation of the documented operator.
    row, col = edge_index[0], edge_index[1]
    query, key, value = jnp.split(conv.lin_qkv(x), 3, axis=-1)
    query_i = query[col].reshape(-1, heads, out_features)
    key_j = key[row].reshape(-1, heads, out_features)
    value_j = value[row].reshape(-1, heads, out_features)
    edge_feat = conv.lin_edge(edge_attr).reshape(-1, heads, out_features)
    alpha = ((query_i * (key_j + edge_feat)).sum(axis=-1)) / jnp.sqrt(out_features)
    alpha = scatter_softmax(alpha, col, dim=0, dim_size=num_nodes)
    messages = ((value_j + edge_feat) * alpha[..., None]).reshape(-1, heads * out_features)
    expected = scatter_add(messages, col, dim_size=num_nodes) + conv.lin_skip(x)

    assert jnp.allclose(out, expected, atol=1e-5)

    # If edge features only entered the values, the output would be affine in edge_attr.
    out_zero = conv(x, edge_index, jnp.zeros_like(edge_attr))
    out_double = conv(x, edge_index, edge_attr * 2.0)
    assert not jnp.allclose(out_double - out_zero, 2.0 * (out - out_zero), atol=1e-4)


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


def test_transformer_conv_no_output_projection():
    """The forward is exactly attention aggregation plus skip -- no outer map.

    The documented operator ends at :math:`W_1 x_i + \\sum_j \\alpha_{ij} W_2 x_j`;
    an extra output projection would make every entry of this hand-computed
    reference wrong by a non-identity linear map.
    """
    heads, out_features = 2, 3
    num_nodes = 4
    x = random.normal(random.key(7), (num_nodes, 5))
    edge_index = jnp.array([[0, 1, 2, 3, 1], [1, 2, 3, 0, 0]])

    conv = TransformerConv(5, out_features, heads, rngs=nnx.Rngs(0))
    out = conv(x, edge_index)

    row, col = edge_index[0], edge_index[1]
    query, key, value = jnp.split(conv.lin_qkv(x), 3, axis=-1)
    query_i = query[col].reshape(-1, heads, out_features)
    key_j = key[row].reshape(-1, heads, out_features)
    value_j = value[row].reshape(-1, heads, out_features)
    alpha = (query_i * key_j).sum(axis=-1) / jnp.sqrt(out_features)
    alpha = scatter_softmax(alpha, col, dim=0, dim_size=num_nodes)
    messages = (value_j * alpha[..., None]).reshape(-1, heads * out_features)
    expected = scatter_add(messages, col, dim_size=num_nodes) + conv.lin_skip(x)

    assert jnp.allclose(out, expected, atol=1e-5)


def test_transformer_conv_qkv_projections_have_bias():
    """The fused q/k/v projection carries a bias, like PyG's three Linears."""
    conv = TransformerConv(5, 3, heads=2, rngs=nnx.Rngs(0))
    assert conv.lin_qkv.use_bias


def test_transformer_conv_beta_requires_root_weight():
    """Without a skip term there is nothing to gate: beta=True is ignored.

    An isolated target node then receives no messages and no root term, so its
    output row is exactly zero; the pre-fix behavior gated against the node's
    raw value projection and produced a nonzero row.
    """
    x = random.normal(random.key(3), (3, 4))
    # Node 2 has no incoming edges.
    edge_index = jnp.array([[0, 1], [1, 0]])

    conv = TransformerConv(4, 6, heads=2, beta=True, root_weight=False, rngs=nnx.Rngs(0))
    assert conv.beta is False
    assert conv.lin_beta is None

    out = conv(x, edge_index)
    assert jnp.allclose(out[2], 0.0)


def test_transformer_conv_skip_width_follows_concat():
    """With concat=False the skip map produces out_features directly.

    Heads are averaged before the skip term, so lin_skip and lin_beta act at
    the final output width.
    """
    conv = TransformerConv(5, 3, heads=2, concat=False, beta=True, rngs=nnx.Rngs(0))
    assert conv.lin_skip.kernel.shape == (5, 3)
    assert conv.lin_beta.kernel.shape == (9, 1)

    x = random.normal(random.key(5), (4, 5))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    out = conv(x, edge_index)
    assert out.shape == (4, 3)


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
