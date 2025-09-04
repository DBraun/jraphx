import jax.numpy as jnp
import jax.random as random
from flax import nnx

from jraphx.nn.norm import GraphNorm


def test_graph_norm():
    """Test basic GraphNorm functionality."""
    # Set seed for reproducibility
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))
    # Create 4 graphs with 50 nodes each
    batch = jnp.repeat(jnp.arange(4), 50)

    norm = GraphNorm(16, rngs=nnx.Rngs(42))
    assert norm.num_features == 16
    assert norm.eps == 1e-5

    # Test without batch (single graph)
    out = norm(x)
    assert out.shape == (200, 16)

    # For single graph, should normalize across all nodes and features
    # The mean should be close to 0 and std close to 1 globally
    global_mean = out.mean()
    global_std = out.std()
    assert jnp.allclose(global_mean, jnp.zeros_like(global_mean), atol=1e-6)
    assert jnp.allclose(global_std, jnp.ones_like(global_std), atol=1e-6)

    # Test with batch (multiple graphs)
    out_batch = norm(x, batch)
    assert out_batch.shape == (200, 16)

    # Each graph should be normalized separately
    for b in range(4):
        mask = batch == b
        graph_out = out_batch[mask]

        # Each graph should have approximately zero mean and unit std globally
        graph_mean = graph_out.mean()
        graph_std = graph_out.std()
        assert jnp.allclose(graph_mean, jnp.zeros_like(graph_mean), atol=1e-6)
        assert jnp.allclose(graph_std, jnp.ones_like(graph_std), atol=1e-6)


def test_graph_norm_single_graph():
    """Test GraphNorm on a single graph without batch."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))

    norm = GraphNorm(16, rngs=nnx.Rngs(42))
    out = norm(x)

    assert out.shape == (200, 16)

    # Check that normalization works correctly
    # Mean should be close to 0, std should be close to 1 (globally across all elements)
    global_mean = out.mean()
    global_var = ((out - global_mean) ** 2).mean()
    global_std = jnp.sqrt(global_var)

    assert jnp.allclose(global_mean, jnp.zeros_like(global_mean), atol=1e-6)
    assert jnp.allclose(global_std, jnp.ones_like(global_std), atol=1e-6)


def test_graph_norm_multiple_graphs():
    """Test GraphNorm with multiple graphs in a batch."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))
    batch = jnp.repeat(jnp.arange(4), 50)

    norm = GraphNorm(16, rngs=nnx.Rngs(42))
    out = norm(x, batch)

    assert out.shape == (200, 16)

    # Each graph should be normalized independently
    for b in range(4):
        mask = batch == b
        graph_out = out[mask]

        # For each graph, the global mean and std should be normalized
        graph_mean = graph_out.mean()
        graph_var = ((graph_out - graph_mean) ** 2).mean()
        graph_std = jnp.sqrt(graph_var)

        assert jnp.allclose(graph_mean, jnp.zeros_like(graph_mean), atol=1e-6)
        assert jnp.allclose(graph_std, jnp.ones_like(graph_std), atol=1e-6)


def test_graph_norm_learnable_parameters():
    """Test that GraphNorm has learnable scale and bias parameters."""
    norm = GraphNorm(16, rngs=nnx.Rngs(42))

    # Check that parameters exist and have correct shapes
    assert norm.weight.value.shape == (16,)
    assert norm.bias.value.shape == (16,)

    # Initial values should be ones for weight, zeros for bias
    assert jnp.allclose(norm.weight.value, jnp.ones(16))
    assert jnp.allclose(norm.bias.value, jnp.zeros(16))


def test_graph_norm_empty_graph():
    """Test GraphNorm behavior with empty graphs in batch."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))
    # Create batch with gap (no nodes for batch 1)
    batch = jnp.concatenate(
        [
            jnp.zeros(50, dtype=jnp.int32),  # Graph 0: 50 nodes
            jnp.full(50, 2, dtype=jnp.int32),  # Graph 2: 50 nodes (skip graph 1)
        ]
    )

    norm = GraphNorm(16, rngs=nnx.Rngs(42))
    out = norm(x, batch)

    assert out.shape == (100, 16)

    # Check graphs 0 and 2 are normalized properly
    for b in [0, 2]:
        mask = batch == b
        if mask.sum() > 0:  # Only check non-empty graphs
            graph_out = out[mask]
            graph_mean = graph_out.mean()
            graph_std = graph_out.std()
            assert jnp.allclose(graph_mean, jnp.zeros_like(graph_mean), atol=1e-6)
            assert jnp.allclose(graph_std, jnp.ones_like(graph_std), atol=1e-6)


def test_graph_norm_consistency():
    """Test that GraphNorm gives consistent results."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))

    norm1 = GraphNorm(16, rngs=nnx.Rngs(42))
    norm2 = GraphNorm(16, rngs=nnx.Rngs(42))

    # Both should give same results with same input
    out1 = norm1(x)
    out2 = norm2(x)

    assert jnp.allclose(out1, out2)


if __name__ == "__main__":
    test_graph_norm()
    test_graph_norm_single_graph()
    test_graph_norm_multiple_graphs()
    test_graph_norm_learnable_parameters()
    test_graph_norm_empty_graph()
    test_graph_norm_consistency()
    print("All GraphNorm tests passed!")
