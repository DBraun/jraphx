import jax
import jax.numpy as jnp
import jax.random as random
import pytest
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

    # Statistics are computed per feature channel over the nodes of the graph
    feature_mean = out.mean(axis=0)
    feature_std = out.std(axis=0)
    assert jnp.allclose(feature_mean, jnp.zeros_like(feature_mean), atol=1e-5)
    assert jnp.allclose(feature_std, jnp.ones_like(feature_std), atol=1e-4)

    # Test with batch (multiple graphs)
    out_batch = norm(x, batch)
    assert out_batch.shape == (200, 16)

    # Each graph should be normalized separately, per feature channel
    for b in range(4):
        mask = batch == b
        graph_out = out_batch[mask]

        graph_mean = graph_out.mean(axis=0)
        graph_std = graph_out.std(axis=0)
        assert jnp.allclose(graph_mean, jnp.zeros_like(graph_mean), atol=1e-5)
        assert jnp.allclose(graph_std, jnp.ones_like(graph_std), atol=1e-4)


def test_graph_norm_single_graph():
    """Test GraphNorm on a single graph without batch."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))

    norm = GraphNorm(16, rngs=nnx.Rngs(42))
    out = norm(x)

    assert out.shape == (200, 16)

    # Each feature channel is centred and scaled independently
    feature_mean = out.mean(axis=0)
    feature_var = ((out - feature_mean) ** 2).mean(axis=0)
    feature_std = jnp.sqrt(feature_var)

    assert jnp.allclose(feature_mean, jnp.zeros_like(feature_mean), atol=1e-5)
    assert jnp.allclose(feature_std, jnp.ones_like(feature_std), atol=1e-4)


def test_graph_norm_multiple_graphs():
    """Test GraphNorm with multiple graphs in a batch."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))
    batch = jnp.repeat(jnp.arange(4), 50)

    norm = GraphNorm(16, rngs=nnx.Rngs(42))
    out = norm(x, batch)

    assert out.shape == (200, 16)

    # Each graph should be normalized independently, per feature channel
    for b in range(4):
        mask = batch == b
        graph_out = out[mask]

        graph_mean = graph_out.mean(axis=0)
        graph_var = ((graph_out - graph_mean) ** 2).mean(axis=0)
        graph_std = jnp.sqrt(graph_var)

        assert jnp.allclose(graph_mean, jnp.zeros_like(graph_mean), atol=1e-5)
        assert jnp.allclose(graph_std, jnp.ones_like(graph_std), atol=1e-4)


def test_graph_norm_per_feature_statistics():
    """Feature channels of very different magnitude are normalized independently."""
    x = jnp.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    batch = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    norm = GraphNorm(2, rngs=nnx.Rngs(0))
    out = norm(x, batch)

    # Each graph has two nodes, so per-column normalization yields exactly -1/+1
    expected = jnp.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]])
    assert jnp.allclose(out, expected, atol=1e-4)

    for b in range(2):
        graph_out = out[batch == b]
        assert jnp.allclose(graph_out.mean(axis=0), jnp.zeros(2), atol=1e-5)
        assert jnp.allclose(graph_out.std(axis=0), jnp.ones(2), atol=1e-4)


def test_graph_norm_learnable_parameters():
    """Test that GraphNorm has learnable scale, bias and mean-scale parameters."""
    norm = GraphNorm(16, rngs=nnx.Rngs(42))

    # Check that parameters exist and have correct shapes
    assert norm.weight[...].shape == (16,)
    assert norm.bias[...].shape == (16,)
    assert norm.mean_scale[...].shape == (16,)

    # Initial values should be ones for weight and mean_scale, zeros for bias
    assert jnp.allclose(norm.weight[...], jnp.ones(16))
    assert jnp.allclose(norm.bias[...], jnp.zeros(16))
    assert jnp.allclose(norm.mean_scale[...], jnp.ones(16))

    # All three must be registered as parameters so optimizers pick them up
    params = nnx.state(norm, nnx.Param)
    assert set(params.keys()) == {"weight", "bias", "mean_scale"}


def test_graph_norm_mean_scale_controls_centering():
    """``mean_scale`` scales how much of the graph mean is subtracted."""
    x = jnp.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    batch = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    norm = GraphNorm(2, rngs=nnx.Rngs(0))
    norm.mean_scale[...] = jnp.zeros(2)
    out = norm(x, batch)

    # With mean_scale == 0 nothing is subtracted; the "variance" is E[x^2]
    graph0 = x[:2]
    expected0 = graph0 / jnp.sqrt((graph0**2).mean(axis=0) + norm.eps)
    assert jnp.allclose(out[:2], expected0, atol=1e-4)

    # The output is no longer zero-mean, unlike the mean_scale == 1 case
    assert not jnp.allclose(out[:2].mean(axis=0), jnp.zeros(2), atol=1e-3)


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
    out = norm(x, batch, batch_size=3)

    assert out.shape == (100, 16)
    assert bool(jnp.all(jnp.isfinite(out)))

    # Check graphs 0 and 2 are normalized properly
    for b in [0, 2]:
        graph_out = out[batch == b]
        graph_mean = graph_out.mean(axis=0)
        graph_std = graph_out.std(axis=0)
        assert jnp.allclose(graph_mean, jnp.zeros_like(graph_mean), atol=1e-5)
        assert jnp.allclose(graph_std, jnp.ones_like(graph_std), atol=1e-4)


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


def test_graph_norm_jit():
    """GraphNorm traces under ``nnx.jit`` when ``batch_size`` is static."""
    key = random.PRNGKey(0)
    x = random.normal(key, (6, 4))
    batch = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)

    norm = GraphNorm(4, rngs=nnx.Rngs(0))

    @nnx.jit
    def run(module: GraphNorm, features: jax.Array, batch_vector: jax.Array) -> jax.Array:
        return module(features, batch_vector, batch_size=2)

    out_jit = run(norm, x, batch)
    out_eager = norm(x, batch)

    assert out_jit.shape == (6, 4)
    assert jnp.allclose(out_jit, out_eager, atol=1e-5)


def test_graph_norm_batch_size_required_under_jit():
    """Without a static ``batch_size`` the traced batch vector cannot be reduced."""
    x = jnp.ones((4, 2))
    batch = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    norm = GraphNorm(2, rngs=nnx.Rngs(0))

    @nnx.jit
    def run(module: GraphNorm, features: jax.Array, batch_vector: jax.Array) -> jax.Array:
        return module(features, batch_vector)

    with pytest.raises(jax.errors.ConcretizationTypeError):
        run(norm, x, batch)


if __name__ == "__main__":
    test_graph_norm()
    test_graph_norm_single_graph()
    test_graph_norm_multiple_graphs()
    test_graph_norm_per_feature_statistics()
    test_graph_norm_learnable_parameters()
    test_graph_norm_mean_scale_controls_centering()
    test_graph_norm_empty_graph()
    test_graph_norm_consistency()
    test_graph_norm_jit()
    test_graph_norm_batch_size_required_under_jit()
    print("All GraphNorm tests passed!")
