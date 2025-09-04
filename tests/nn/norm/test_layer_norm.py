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
    """Test LayerNorm in graph mode specifically."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))
    # Create 4 graphs with 50 nodes each
    batch = jnp.repeat(jnp.arange(4), 50)

    norm = LayerNorm(16, mode="graph", rngs=nnx.Rngs(42))
    out = norm(x, batch)

    # In graph mode, each graph should be normalized as a unit
    # For each graph, nodes should have consistent normalization
    for b in range(4):
        mask = batch == b
        graph_out = out[mask]

        # Each node in the graph should have the same mean and std pattern
        node_means = graph_out.mean(axis=1)
        node_stds = graph_out.std(axis=1)

        # All nodes in the same graph should have similar statistics
        assert jnp.allclose(node_means, jnp.zeros_like(node_means), atol=1e-6)
        assert jnp.allclose(node_stds, jnp.ones_like(node_stds), atol=1e-4)


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
    assert jnp.allclose(norm.weight.value, jnp.ones(16))
    assert jnp.allclose(norm.bias.value, jnp.zeros(16))

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
    test_layer_norm_no_affine()
    test_layer_norm_with_affine()
    test_layer_norm_multi_dimensional()

    print("All LayerNorm tests passed!")
