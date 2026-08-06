"""Test cases for the JraphX GATv2Conv layer."""

import numpy as np
import pytest
from flax import nnx
from jax import nn as jnn
from jax import numpy as jnp
from jax import random

from jraphx.nn.conv import GATv2Conv
from jraphx.utils import scatter_add, scatter_softmax


def test_gatv2_conv_basic():
    """Test basic GATv2 convolution functionality."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = GATv2Conv(8, 32, heads=2, rngs=nnx.Rngs(0))
    out = conv(x, edge_index)
    assert out.shape == (4, 64)  # heads * out_features when concat=True


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_gatv2_conv_heads(heads):
    """Test GATv2 with different numbers of attention heads."""
    x = random.normal(random.key(42), (5, 10))
    edge_index = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

    conv = GATv2Conv(10, 20, heads=heads, rngs=nnx.Rngs(0))
    assert conv(x, edge_index).shape == (5, heads * 20)

    conv_avg = GATv2Conv(10, 20, heads=heads, concat=False, rngs=nnx.Rngs(0))
    assert conv_avg(x, edge_index).shape == (5, 20)


def test_gatv2_conv_softmax_is_per_head():
    """Attention coefficients must sum to one per (target node, head), not jointly."""
    heads = 4
    x = random.normal(random.key(7), (5, 6))
    # Nodes 1 and 3 each receive three incoming edges.
    edge_index = jnp.array([[0, 2, 4, 0, 1, 2], [1, 1, 1, 3, 3, 3]])

    conv = GATv2Conv(6, 3, heads=heads, add_self_loops=False, rngs=nnx.Rngs(0))
    _, (returned_edge_index, alpha) = conv(x, edge_index, return_attention_weights=True)

    assert alpha.shape == (6, heads)
    assert jnp.array_equal(returned_edge_index, edge_index)

    target = np.asarray(edge_index[1])
    for node in [1, 3]:
        per_head_sum = np.asarray(alpha)[target == node].sum(axis=0)
        # Each head is normalized independently ...
        assert np.allclose(per_head_sum, np.ones(heads), atol=1e-5)
        # ... so the joint sum over all heads is `heads`, not 1.
        assert np.isclose(per_head_sum.sum(), heads, atol=1e-5)


def test_gatv2_conv_attention_matches_reference():
    """The layer reproduces the GATv2 operator edge by edge."""
    heads, out_features = 3, 4
    x = random.normal(random.key(11), (6, 5))
    edge_index = jnp.array([[0, 1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 0, 0]])

    conv = GATv2Conv(5, out_features, heads=heads, add_self_loops=False, rngs=nnx.Rngs(3))
    out, (_, alpha) = conv(x, edge_index, return_attention_weights=True)

    num_nodes = x.shape[0]
    row, col = edge_index[0], edge_index[1]
    x_l = conv.lin_l(x).reshape(num_nodes, heads, out_features)
    x_r = conv.lin_r(x).reshape(num_nodes, heads, out_features)
    combined = jnn.leaky_relu(x_l[row] + x_r[col], negative_slope=conv.negative_slope)
    scores = jnp.sum(combined * conv.att[...], axis=-1)
    expected_alpha = scatter_softmax(scores, col, dim_size=num_nodes)
    messages = (x_l[row] * expected_alpha[..., None]).reshape(-1, heads * out_features)
    expected_out = scatter_add(messages, col, dim_size=num_nodes) + conv.bias[...]

    assert jnp.allclose(alpha, expected_alpha, atol=1e-6)
    assert jnp.allclose(out, expected_out, atol=1e-5)


def test_gatv2_conv_edge_features():
    """Test GATv2 with edge features of different dimensionalities."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_attr = random.normal(random.key(456), (edge_index.shape[1], 4))

    conv = GATv2Conv(8, 16, heads=2, edge_dim=4, rngs=nnx.Rngs(0))
    out = conv(x, edge_index, edge_attr)
    assert out.shape == (4, 32)

    out_other = conv(x, edge_index, edge_attr * 2.0)
    assert not jnp.allclose(out, out_other)


def test_gatv2_conv_share_weights():
    """`share_weights=True` reuses a single linear map for source and target."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    conv = GATv2Conv(8, 16, heads=2, share_weights=True, rngs=nnx.Rngs(0))
    assert conv.lin_l is conv.lin_r
    assert conv(x, edge_index).shape == (4, 32)


def test_gatv2_conv_deterministic():
    """Test that GATv2 is deterministic with same inputs and RNG."""
    conv1 = GATv2Conv(6, 12, heads=2, rngs=nnx.Rngs(42))
    conv2 = GATv2Conv(6, 12, heads=2, rngs=nnx.Rngs(42))

    x = jnp.ones((4, 6))
    edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    assert jnp.allclose(conv1(x, edge_index), conv2(x, edge_index))


def test_gatv2_conv_empty_edges():
    """Test GATv2 with an empty edge index."""
    x = random.normal(random.key(42), (4, 8))
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)

    conv = GATv2Conv(8, 32, heads=2, rngs=nnx.Rngs(0))
    assert conv(x, edge_index).shape == (4, 64)


def test_gatv2_conv_does_not_duplicate_existing_self_loops():
    """A pre-existing self-loop must not be counted twice in the attention softmax.

    PyG removes self-loops before inserting its own; see the GATConv counterpart.
    """
    x = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    with_loop = jnp.array([[0, 1, 0], [0, 0, 1]])  # (0, 0) already present
    without_loop = jnp.array([[1, 0], [0, 1]])  # same graph, loop removed

    conv = GATv2Conv(2, 2, heads=1, rngs=nnx.Rngs(0))
    assert jnp.allclose(conv(x, with_loop), conv(x, without_loop), atol=1e-6)


def test_gatv2_conv_bipartite_self_loops_use_the_smaller_node_count():
    """Self-loops only exist for nodes present in both endpoint tables."""
    x_src = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    x_dst = jnp.ones((4, 2))
    edge_index = jnp.array([[0, 1], [0, 3]])

    conv = GATv2Conv((2, 2), 3, heads=1, bias=False, rngs=nnx.Rngs(0))
    out = conv((x_src, x_dst), edge_index)

    assert out.shape == (4, 3)
    assert bool(jnp.isfinite(out).all())
    assert jnp.allclose(out[2], 0.0)
    assert not jnp.allclose(out[1], out[2])


def test_gatv2_conv_rejects_out_of_range_source_index():
    """An index genuinely past the source table must raise, not silently clamp."""
    x_src = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    x_dst = jnp.ones((4, 2))
    conv = GATv2Conv((2, 2), 3, heads=1, rngs=nnx.Rngs(0))

    with pytest.raises(IndexError, match="Source indices"):
        conv((x_src, x_dst), jnp.array([[0, 7], [0, 3]]))
