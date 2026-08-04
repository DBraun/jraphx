import jax.numpy as jnp
import jax.random as random
import pytest
from flax import nnx

from jraphx.nn.norm import LayerNorm


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("mode", ["graph", "node"])
def test_layer_norm(affine, mode):
    """Test LayerNorm functionality with different configurations."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))
    batch = jnp.zeros(100, dtype=jnp.int32)

    norm = LayerNorm(
        16, elementwise_affine=affine, mode=mode, rngs=nnx.Rngs(42) if affine else None
    )
    assert norm.eps == 1e-5
    assert norm.elementwise_affine == affine
    assert norm.mode == mode

    # Test without batch
    out1 = norm(x)
    assert out1.shape == (100, 16)

    # Test with batch (should be equivalent for single graph)
    out_with_batch = norm(x, batch)
    assert jnp.allclose(out_with_batch, out1, atol=1e-6)

    # Test with multiple graphs
    batch_multi = jnp.concatenate([jnp.zeros(100, dtype=jnp.int32), jnp.ones(100, dtype=jnp.int32)])
    x_multi = jnp.concatenate([x, x], axis=0)

    out2 = norm(x_multi, batch_multi)
    assert out2.shape == (200, 16)

    # For graph mode, each graph should be normalized separately
    if mode == "graph":
        assert jnp.allclose(out1, out2[:100], atol=1e-6)
        assert jnp.allclose(out1, out2[100:], atol=1e-6)


def test_layer_norm_node_mode():
    """Test LayerNorm in node mode specifically."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))

    norm = LayerNorm(16, mode="node", rngs=nnx.Rngs(42))
    out = norm(x)

    # In node mode, each node should be normalized independently
    # Check that each node has approximately zero mean and unit variance
    node_means = out.mean(axis=1)
    node_stds = out.std(axis=1)

    assert jnp.allclose(node_means, jnp.zeros_like(node_means), atol=1e-6)
    assert jnp.allclose(node_stds, jnp.ones_like(node_stds), atol=1e-4)


def test_layer_norm_graph_mode():
    """Graph mode reduces over both the node axis and the feature axis."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16)) * jnp.arange(1, 17) + jnp.arange(16)
    # Create 4 graphs with 50 nodes each
    batch = jnp.repeat(jnp.arange(4), 50)

    norm = LayerNorm(16, mode="graph", rngs=nnx.Rngs(42))
    out = norm(x, batch)

    # The whole (nodes x features) block of each graph is one normalized unit
    for b in range(4):
        graph_out = out[batch == b]
        assert jnp.allclose(graph_out.mean(), 0.0, atol=1e-5)
        assert jnp.allclose(graph_out.std(), 1.0, atol=1e-4)

    # Individual nodes are NOT separately normalized in graph mode
    node_means = out.mean(axis=1)
    assert not jnp.allclose(node_means, jnp.zeros_like(node_means), atol=1e-3)


def test_layer_norm_graph_mode_differs_from_node_mode():
    """Graph mode and node mode must not produce identical output."""
    x = jnp.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    batch = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    node_out = LayerNorm(2, mode="node", rngs=nnx.Rngs(42))(x, batch)
    graph_out = LayerNorm(2, mode="graph", rngs=nnx.Rngs(42))(x, batch)

    assert not jnp.allclose(node_out, graph_out)

    # Node mode: every row is normalized on its own two values -> [-1, +1]
    assert jnp.allclose(node_out, jnp.tile(jnp.array([-1.0, 1.0]), (4, 1)), atol=1e-4)

    # Graph mode: one scalar mean/std per graph over the 2x2 block
    block = x[:2]
    expected = (block - block.mean()) / jnp.sqrt(block.var() + 1e-5)
    assert jnp.allclose(graph_out[:2], expected, atol=1e-4)


def test_layer_norm_graph_mode_no_batch():
    """Graph mode without a batch vector normalizes the whole input as one graph."""
    key = random.PRNGKey(7)
    x = random.normal(key, (30, 8)) * 3.0 + 2.0

    norm = LayerNorm(8, mode="graph", rngs=nnx.Rngs(0))
    out_no_batch = norm(x)
    out_single_graph = norm(x, jnp.zeros(30, dtype=jnp.int32))

    assert jnp.allclose(out_no_batch, out_single_graph, atol=1e-6)
    assert jnp.allclose(out_no_batch.mean(), 0.0, atol=1e-5)
    assert jnp.allclose(out_no_batch.std(), 1.0, atol=1e-4)


@pytest.mark.parametrize("use_bias,use_scale", [(False, True), (True, False), (False, False)])
def test_layer_norm_graph_mode_partial_affine(use_bias, use_scale):
    """Graph mode must not dereference the affine parameters that were not created."""
    key = random.PRNGKey(3)
    x = random.normal(key, (10, 4))
    batch = jnp.array([0] * 6 + [1] * 4, dtype=jnp.int32)

    norm = LayerNorm(4, mode="graph", use_bias=use_bias, use_scale=use_scale, rngs=nnx.Rngs(0))
    assert (norm.weight is not None) == use_scale
    assert (norm.bias is not None) == use_bias

    out = norm(x, batch)
    assert out.shape == (10, 4)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_layer_norm_graph_mode_dtype():
    """Graph mode honours the ``dtype`` argument, like node mode."""
    key = random.PRNGKey(11)
    x = random.normal(key, (8, 4))
    batch = jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)

    norm = LayerNorm(4, mode="graph", dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    out = norm(x, batch)
    assert out.dtype == jnp.bfloat16


def test_layer_norm_graph_mode_empty_graph():
    """An empty segment must not produce NaNs in the other graphs."""
    key = random.PRNGKey(5)
    x = random.normal(key, (8, 4))
    batch = jnp.array([0, 0, 0, 0, 2, 2, 2, 2], dtype=jnp.int32)

    norm = LayerNorm(4, mode="graph", rngs=nnx.Rngs(0))
    out = norm(x, batch, batch_size=3)

    assert bool(jnp.all(jnp.isfinite(out)))
    for b in [0, 2]:
        graph_out = out[batch == b]
        assert jnp.allclose(graph_out.mean(), 0.0, atol=1e-5)
        assert jnp.allclose(graph_out.std(), 1.0, atol=1e-4)


def test_layer_norm_graph_mode_preserves_mask_argument():
    """The ``mask`` keyword is not clobbered by the graph-mode reduction."""
    key = random.PRNGKey(13)
    x = random.normal(key, (6, 4))
    batch = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)

    norm = LayerNorm(4, mode="graph", rngs=nnx.Rngs(0))
    node_mask = jnp.ones(6, dtype=bool)

    assert jnp.allclose(norm(x, batch, mask=node_mask), norm(x, batch))


def test_layer_norm_graph_mode_jit():
    """Graph mode traces under ``nnx.jit`` when ``batch_size`` is static."""
    key = random.PRNGKey(17)
    x = random.normal(key, (6, 4))
    batch = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)

    norm = LayerNorm(4, mode="graph", rngs=nnx.Rngs(0))

    @nnx.jit
    def run(module: LayerNorm, features: jnp.ndarray, batch_vector: jnp.ndarray) -> jnp.ndarray:
        return module(features, batch_vector, batch_size=2)

    assert jnp.allclose(run(norm, x, batch), norm(x, batch), atol=1e-5)


def test_layer_norm_unknown_mode():
    """An unrecognized mode is reported instead of silently falling back."""
    key = random.PRNGKey(19)
    x = random.normal(key, (4, 4))

    norm = LayerNorm(4, mode="element", rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="Unknown LayerNorm mode"):
        norm(x)


def test_layer_norm_no_affine():
    """Test LayerNorm without learnable parameters."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))

    norm = LayerNorm(16, elementwise_affine=False)
    assert norm.weight is None
    assert norm.bias is None

    out = norm(x)
    assert out.shape == (100, 16)

    # Check normalization properties
    node_means = out.mean(axis=1)
    node_stds = out.std(axis=1)

    assert jnp.allclose(node_means, jnp.zeros_like(node_means), atol=1e-6)
    assert jnp.allclose(node_stds, jnp.ones_like(node_stds), atol=1e-4)


def test_layer_norm_with_affine():
    """Test LayerNorm with learnable parameters."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))

    norm = LayerNorm(16, elementwise_affine=True, rngs=nnx.Rngs(42))
    assert norm.weight is not None
    assert norm.bias is not None

    # Initial weight should be ones, bias should be zeros
    assert jnp.allclose(norm.weight[...], jnp.ones(16))
    assert jnp.allclose(norm.bias[...], jnp.zeros(16))

    out = norm(x)
    assert out.shape == (100, 16)


def test_layer_norm_multi_dimensional():
    """Test LayerNorm with multi-dimensional features."""
    key = random.PRNGKey(42)
    x = random.normal(key, (50, 8, 4))  # 50 nodes, 8x4 features

    # Test with list of dimensions
    norm = LayerNorm([8, 4], rngs=nnx.Rngs(42))
    assert norm.normalized_shape == (8, 4)

    out = norm(x)
    assert out.shape == (50, 8, 4)


# TODO: HeteroLayerNorm is not implemented in JraphX
# The following test from PyG cannot be converted:
# - test_hetero_layer_norm: Requires HeteroLayerNorm which handles different node types


if __name__ == "__main__":
    # Run basic tests
    test_layer_norm(True, "node")
    test_layer_norm(False, "node")
    test_layer_norm(True, "graph")
    test_layer_norm(False, "graph")

    test_layer_norm_node_mode()
    test_layer_norm_graph_mode()
    test_layer_norm_graph_mode_differs_from_node_mode()
    test_layer_norm_graph_mode_no_batch()
    test_layer_norm_graph_mode_partial_affine(False, True)
    test_layer_norm_graph_mode_partial_affine(True, False)
    test_layer_norm_graph_mode_partial_affine(False, False)
    test_layer_norm_graph_mode_dtype()
    test_layer_norm_graph_mode_empty_graph()
    test_layer_norm_graph_mode_preserves_mask_argument()
    test_layer_norm_graph_mode_jit()
    test_layer_norm_unknown_mode()
    test_layer_norm_no_affine()
    test_layer_norm_with_affine()
    test_layer_norm_multi_dimensional()

    print("All LayerNorm tests passed!")
