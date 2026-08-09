"""Test cases for JraphX GCNConv layer converted from PyTorch Geometric tests."""

import jax
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
    assert conv_no_bias.bias is None

    # The bias lives outside the linear layer so that it is not aggregated
    conv_bias = GCNConv(8, 16, bias=True, rngs=nnx.Rngs(0))
    assert conv_bias.linear.use_bias is False
    assert conv_bias.bias[...].shape == (16,)
    assert jnp.allclose(conv_bias.bias[...], jnp.zeros(16))

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


def test_gcn_norm_uses_weighted_degree():
    """Normalization must divide by the weighted degree, matching PyG."""
    conv = GCNConv(3, 3, rngs=nnx.Rngs(0))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    edge_weight = jnp.array([2.0, 2.0, 0.5, 0.5, 3.0])

    out_index, out_weight = conv.gcn_norm(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=4,
    )

    # Self-loops appended for all four nodes
    assert out_index.shape == (2, 9)

    # Weighted in-degrees (including unit self-loops): [6, 3.5, 1.5, 1]
    deg = jnp.array([6.0, 3.5, 1.5, 1.0])
    deg_inv_sqrt = deg ** (-0.5)
    weights = jnp.concatenate([edge_weight, jnp.ones(4)])
    expected = deg_inv_sqrt[out_index[0]] * weights * deg_inv_sqrt[out_index[1]]

    assert jnp.allclose(out_weight, expected, atol=1e-6)
    # Pinned against torch_geometric's gcn_norm for the same inputs
    assert jnp.allclose(
        out_weight,
        jnp.array(
            [0.436436, 0.436436, 0.218218, 0.218218, 1.224745, 0.166667, 0.285714, 0.666667, 1.0]
        ),
        atol=1e-5,
    )


def test_gcn_norm_improved_fill_value_enters_degree():
    """With improved=True the fill value of 2.0 must also count in the degree."""
    conv = GCNConv(3, 3, improved=True, rngs=nnx.Rngs(0))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])

    out_index, out_weight = conv.gcn_norm(
        edge_index=edge_index,
        num_nodes=4,
        improved=True,
    )

    assert jnp.allclose(
        out_weight,
        jnp.array([0.25, 0.25, 0.288675, 0.288675, 0.353553, 0.5, 0.5, 0.666667, 1.0]),
        atol=1e-5,
    )

    # Node 3 only has its own self-loop, so its coefficient is exactly 1.0
    assert float(out_weight[-1]) == pytest.approx(1.0, abs=1e-6)

    # No coefficient may exceed 1 for a proper normalization
    assert bool(jnp.all(out_weight <= 1.0 + 1e-6))


def test_gcn_norm_isolated_node_gets_zero_weight():
    """A node with a non-positive weighted degree receives zero normalization."""
    conv = GCNConv(3, 3, rngs=nnx.Rngs(0))
    edge_index = jnp.array([[0], [1]])
    edge_weight = jnp.array([1.0])

    _, out_weight = conv.gcn_norm(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=3,
        add_self_loops=False,
    )

    # Node 0 has zero in-degree, so the single edge is scaled to zero
    assert jnp.allclose(out_weight, jnp.zeros(1))


def test_gcn_norm_does_not_duplicate_existing_self_loops():
    """An existing self-loop keeps its weight and enters the degree only once."""
    conv = GCNConv(3, 3, rngs=nnx.Rngs(0))
    edge_index = jnp.array([[0, 1, 1], [1, 0, 1]])
    edge_weight = jnp.array([2.0, 2.0, 5.0])

    out_index, out_weight = conv.gcn_norm(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=2,
    )

    # Exactly one effective loop per node: the pre-existing entry is zeroed and
    # its weight of 5.0 moves to the appended loop of node 1.
    is_loop = out_index[0] == out_index[1]
    assert int(jnp.sum(jnp.abs(out_weight[is_loop]) > 0)) == 2

    # Weighted in-degrees are [2 + 1, 2 + 5], as in torch_geometric's gcn_norm
    deg_inv_sqrt = jnp.array([3.0, 7.0]) ** (-0.5)
    expected = jnp.array(
        [
            deg_inv_sqrt[0] * 2.0 * deg_inv_sqrt[1],
            deg_inv_sqrt[1] * 2.0 * deg_inv_sqrt[0],
            0.0,
            deg_inv_sqrt[0] * 1.0 * deg_inv_sqrt[0],
            deg_inv_sqrt[1] * 5.0 * deg_inv_sqrt[1],
        ]
    )
    assert jnp.allclose(out_weight, expected, atol=1e-6)
    # Pinned against torch_geometric's gcn_norm for the same inputs
    assert jnp.allclose(
        out_weight[out_weight > 0],
        jnp.array([0.436436, 0.436436, 0.333333, 0.714286]),
        atol=1e-5,
    )


def test_gcn_norm_improved_keeps_existing_self_loop_weight():
    """With improved=True only nodes without a loop receive the fill value 2.0."""
    conv = GCNConv(3, 3, improved=True, rngs=nnx.Rngs(0))
    edge_index = jnp.array([[0, 1, 1], [1, 0, 1]])
    edge_weight = jnp.array([2.0, 2.0, 5.0])

    out_index, out_weight = conv.gcn_norm(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=2,
        improved=True,
    )

    # Degrees are [2 + 2, 2 + 5]: node 0 gets the fill value, node 1 keeps 5.0
    deg_inv_sqrt = jnp.array([4.0, 7.0]) ** (-0.5)
    expected = jnp.array(
        [
            deg_inv_sqrt[0] * 2.0 * deg_inv_sqrt[1],
            deg_inv_sqrt[1] * 2.0 * deg_inv_sqrt[0],
            0.0,
            deg_inv_sqrt[0] * 2.0 * deg_inv_sqrt[0],
            deg_inv_sqrt[1] * 5.0 * deg_inv_sqrt[1],
        ]
    )
    assert jnp.allclose(out_weight, expected, atol=1e-6)
    assert out_index.shape == (2, 5)


def test_gcn_norm_is_jittable_with_existing_self_loops():
    """Self-loop insertion keeps a static shape, so gcn_norm stays traceable."""
    conv = GCNConv(3, 3, rngs=nnx.Rngs(0))
    edge_index = jnp.array([[0, 1, 1], [1, 0, 1]])
    edge_weight = jnp.array([2.0, 2.0, 5.0])

    jitted = jax.jit(lambda ei, ew: conv.gcn_norm(ei, ew, 2))
    jit_index, jit_weight = jitted(edge_index, edge_weight)
    eager_index, eager_weight = conv.gcn_norm(edge_index, edge_weight, 2)

    assert jnp.array_equal(jit_index, eager_index)
    assert jnp.allclose(jit_weight, eager_weight, atol=1e-6)


def test_gcn_conv_unit_self_loop_is_a_no_op():
    """Adding a unit-weight self-loop to the input cannot change the output."""
    x = random.normal(random.key(3), (4, 3))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    edge_weight = jnp.array([2.0, 2.0, 0.5, 0.5, 3.0])

    looped_index = jnp.concatenate([edge_index, jnp.array([[2], [2]])], axis=1)
    looped_weight = jnp.concatenate([edge_weight, jnp.array([1.0])])

    conv = GCNConv(3, 2, rngs=nnx.Rngs(0))
    out = conv(x, edge_index, edge_weight)
    out_looped = conv(x, looped_index, looped_weight)

    assert jnp.allclose(out, out_looped, atol=1e-5)


def test_gcn_conv_bias_is_constant_offset():
    """The bias must be a per-row constant, not scaled by the aggregation."""
    x = random.normal(random.key(0), (4, 3))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])

    conv = GCNConv(3, 2, rngs=nnx.Rngs(0))
    out_zero_bias = conv(x, edge_index)

    bias = jnp.array([5.0, -3.0])
    conv.bias[...] = bias
    out_bias = conv(x, edge_index)

    delta = out_bias - out_zero_bias
    expected = jnp.broadcast_to(bias, delta.shape)
    assert jnp.allclose(delta, expected, atol=1e-5)


def test_gcn_conv_bias_constant_without_normalization():
    """The bias offset stays constant on the unweighted propagate path too."""
    x = random.normal(random.key(1), (4, 3))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])

    conv = GCNConv(3, 2, normalize=False, add_self_loops=False, rngs=nnx.Rngs(0))
    out_zero_bias = conv(x, edge_index)

    bias = jnp.array([5.0, -3.0])
    conv.bias[...] = bias
    out_bias = conv(x, edge_index)

    assert jnp.allclose(out_bias - out_zero_bias, jnp.broadcast_to(bias, (4, 2)), atol=1e-5)


def test_gcn_conv_no_bias_has_no_offset():
    """With bias=False the layer adds nothing after aggregation."""
    x = random.normal(random.key(2), (4, 3))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])

    conv = GCNConv(3, 2, bias=False, rngs=nnx.Rngs(0))
    out = conv(x, edge_index)

    assert conv.bias is None
    assert out.shape == (4, 2)


def test_gcn_conv_cached_requires_precompute():
    """A cached layer refuses to run before its cache has been filled."""
    x = jnp.ones((4, 4))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    conv = GCNConv(4, 4, cached=True, rngs=nnx.Rngs(0))

    with pytest.raises(RuntimeError, match="normalization cache is empty"):
        conv(x, edge_index)


def test_precompute_norm_requires_cached():
    """precompute_norm is meaningless without cached=True."""
    conv = GCNConv(4, 4, rngs=nnx.Rngs(0))
    edge_index = jnp.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="requires 'cached=True'"):
        conv.precompute_norm(edge_index, num_nodes=2)


def test_gcn_conv_cached_matches_uncached():
    """A precomputed cache yields exactly the uncached result."""
    x = random.normal(random.key(3), (4, 4))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    edge_weight = jnp.array([2.0, 2.0, 0.5, 0.5, 3.0])

    conv_cached = GCNConv(4, 4, cached=True, rngs=nnx.Rngs(0))
    conv_plain = GCNConv(4, 4, cached=False, rngs=nnx.Rngs(0))

    conv_cached.precompute_norm(edge_index, edge_weight, num_nodes=4, dtype=x.dtype)

    out_cached = conv_cached(x, edge_index, edge_weight)
    out_plain = conv_plain(x, edge_index, edge_weight)

    assert jnp.allclose(out_cached, out_plain, atol=1e-6)

    # Repeated eager calls reuse the same cache entry
    assert jnp.allclose(conv_cached(x, edge_index, edge_weight), out_cached)


def test_gcn_conv_cached_normalizes_only_once():
    """gcn_norm runs once during precompute and never during forward passes."""
    x = jnp.ones((4, 4))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    conv = GCNConv(4, 4, cached=True, rngs=nnx.Rngs(0))

    calls = []
    original_gcn_norm = conv.gcn_norm

    def counting_gcn_norm(*args, **kwargs):
        calls.append(1)
        return original_gcn_norm(*args, **kwargs)

    conv.gcn_norm = counting_gcn_norm
    conv.precompute_norm(edge_index, num_nodes=4, dtype=x.dtype)
    assert len(calls) == 1

    for _ in range(3):
        conv(x, edge_index)
    assert len(calls) == 1


def test_gcn_conv_cached_under_nnx_jit():
    """A precomputed cached layer survives repeated jitted calls."""
    x = jnp.ones((4, 4))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    conv = GCNConv(4, 4, cached=True, rngs=nnx.Rngs(0))
    conv.precompute_norm(edge_index, num_nodes=4, dtype=x.dtype)

    jitted = nnx.jit(lambda m, x, ei: m(x, ei))

    outs = [jitted(conv, x, edge_index) for _ in range(3)]
    for out in outs:
        assert out.shape == (4, 4)
        assert jnp.allclose(out, outs[0])
    assert jnp.allclose(outs[0], conv(x, edge_index), atol=1e-6)


def test_gcn_conv_cached_under_jax_jit():
    """A precomputed cached layer also works under a plain jax.jit closure."""
    x = jnp.ones((4, 4))
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    conv = GCNConv(4, 4, cached=True, rngs=nnx.Rngs(0))
    conv.precompute_norm(edge_index, num_nodes=4, dtype=x.dtype)

    jitted = jax.jit(lambda x, ei: conv(x, ei))

    out1 = jitted(x, edge_index)
    out2 = jitted(x, edge_index)
    assert jnp.allclose(out1, out2)
    assert jnp.allclose(out1, conv(x, edge_index), atol=1e-6)


def test_gcn_conv_cached_rejects_size_change():
    """Feeding a differently sized graph to a cached layer raises."""
    edge_index = jnp.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 0]])
    conv = GCNConv(4, 4, cached=True, rngs=nnx.Rngs(0))
    conv.precompute_norm(edge_index, num_nodes=4)

    with pytest.raises(RuntimeError, match="cached a normalization for 4 nodes"):
        conv(jnp.ones((5, 4)), edge_index)

    conv.reset_cache()
    with pytest.raises(RuntimeError, match="normalization cache is empty"):
        conv(jnp.ones((4, 4)), edge_index)


def test_gcn_conv_cached_dtype_follows_precompute():
    """The cached normalization keeps the dtype requested at precompute time."""
    edge_index = jnp.array([[0, 1], [1, 0]])
    conv = GCNConv(3, 3, cached=True, rngs=nnx.Rngs(0))
    conv.precompute_norm(edge_index, num_nodes=2, dtype=jnp.bfloat16)

    assert conv._cached_edge_weight.get_value().dtype == jnp.bfloat16


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
