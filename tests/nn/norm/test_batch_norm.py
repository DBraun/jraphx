import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from flax import nnx

from jraphx.nn.norm import BatchNorm


def test_batch_norm():
    """Test basic BatchNorm functionality with different configurations."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))

    # Test with affine=True, track_running_stats=True (default)
    norm = BatchNorm(16, rngs=nnx.Rngs(42))
    assert norm.num_features == 16
    assert norm.eps == 1e-5
    assert norm.momentum == 0.99
    assert norm.track_running_stats is True
    assert norm.use_scale is True  # corresponds to affine in PyG
    assert norm.use_bias is True  # corresponds to affine in PyG

    out = norm(x)
    assert out.shape == (100, 16)

    # Test with affine=False, track_running_stats=False
    norm_no_affine = BatchNorm(
        16, track_running_stats=False, use_scale=False, use_bias=False, rngs=nnx.Rngs(42)
    )
    assert norm_no_affine.track_running_stats is False
    assert norm_no_affine.use_scale is False
    assert norm_no_affine.use_bias is False

    out_no_affine = norm_no_affine(x)
    assert out_no_affine.shape == (100, 16)


def test_batch_norm_single_element():
    """Test BatchNorm with single element - should work in JAX unlike PyTorch."""
    key = random.PRNGKey(42)
    x = random.normal(key, (1, 16))

    # JraphX BatchNorm should handle single elements without issues
    norm = BatchNorm(16, rngs=nnx.Rngs(42))
    out = norm(x)
    assert out.shape == (1, 16)

    # Test with track_running_stats=False
    norm_no_stats = BatchNorm(16, track_running_stats=False, rngs=nnx.Rngs(42))
    out_no_stats = norm_no_stats(x)
    assert out_no_stats.shape == (1, 16)


def test_batch_norm_statistics():
    """Test that BatchNorm produces normalized outputs."""
    key = random.PRNGKey(42)
    x = random.normal(key, (100, 16))

    norm = BatchNorm(16, track_running_stats=False, rngs=nnx.Rngs(42))
    out = norm(x)

    # Check normalization properties
    mean = out.mean(axis=0)
    std = out.std(axis=0)

    # Mean should be close to 0, std should be close to 1
    assert jnp.allclose(mean, jnp.zeros_like(mean), atol=1e-6)
    assert jnp.allclose(std, jnp.ones_like(std), atol=1e-6)


def test_batch_norm_running_stats():
    """Test running statistics functionality."""
    key = random.PRNGKey(42)
    x1 = random.normal(key, (100, 16))
    x2 = random.normal(random.split(key)[1], (100, 16))

    norm = BatchNorm(16, track_running_stats=True, rngs=nnx.Rngs(42))

    # Initial running stats should be zeros and ones
    assert jnp.allclose(norm.running_mean[...], jnp.zeros(16))
    assert jnp.allclose(norm.running_var[...], jnp.ones(16))
    assert norm.num_batches_tracked[...] == 0

    # Process first batch
    _ = norm(x1, use_running_average=False)
    assert norm.num_batches_tracked[...] == 1

    # Process second batch
    _ = norm(x2, use_running_average=False)
    assert norm.num_batches_tracked[...] == 2

    # Running stats should have been updated
    assert not jnp.allclose(norm.running_mean[...], jnp.zeros(16))
    assert not jnp.allclose(norm.running_var[...], jnp.ones(16))


def test_batch_norm_with_batch():
    """The batch vector is ignored: statistics are pooled over the whole mini-batch."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))
    # Graphs of deliberately unequal size, so an unweighted mean-of-means differs
    batch = jnp.concatenate(
        [
            jnp.zeros(150, dtype=jnp.int32),
            jnp.ones(40, dtype=jnp.int32),
            jnp.full(10, 2, dtype=jnp.int32),
        ]
    )

    norm = BatchNorm(16, track_running_stats=False, rngs=nnx.Rngs(42))
    out = norm(x, batch)

    assert out.shape == (200, 16)
    assert jnp.allclose(out, norm(x))

    overall_mean = out.mean(axis=0)
    overall_std = out.std(axis=0)
    assert jnp.allclose(overall_mean, jnp.zeros_like(overall_mean), atol=1e-5)
    assert jnp.allclose(overall_std, jnp.ones_like(overall_std), atol=1e-4)


def test_batch_norm_unequal_graphs_are_not_reweighted():
    """A single large-valued graph must not be down-weighted by graph count."""
    x = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [0.0, 0.0]])
    batch = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)

    norm = BatchNorm(2, track_running_stats=False, rngs=nnx.Rngs(0))
    out = norm(x, batch)

    expected = (x - x.mean(axis=0)) / jnp.sqrt(x.var(axis=0) + norm.eps)
    assert jnp.allclose(out, expected, atol=1e-5)
    assert jnp.allclose(out.mean(axis=0), jnp.zeros(2), atol=1e-5)
    assert jnp.allclose(out.var(axis=0), jnp.ones(2), atol=1e-4)


def test_batch_norm_jit_with_batch():
    """BatchNorm traces under ``nnx.jit``, including when a batch vector is passed."""
    key = random.PRNGKey(0)
    x = random.normal(key, (12, 4))
    batch = jnp.array([0] * 7 + [1] * 5, dtype=jnp.int32)

    norm = BatchNorm(4, track_running_stats=True, rngs=nnx.Rngs(0))

    @nnx.jit
    def run(module: BatchNorm, features: jax.Array, batch_vector: jax.Array) -> jax.Array:
        return module(features, batch_vector)

    out = run(norm, x, batch)
    assert out.shape == (12, 4)
    assert norm.num_batches_tracked[...] == 1
    assert bool(jnp.all(jnp.isfinite(out)))


def test_batch_norm_running_stats_are_batch_stats():
    """Running statistics split out under the standard ``nnx.BatchStat`` filter."""
    norm = BatchNorm(8, track_running_stats=True, rngs=nnx.Rngs(0))

    params = nnx.state(norm, nnx.Param)
    batch_stats = nnx.state(norm, nnx.BatchStat)

    assert set(params.keys()) == {"weight", "bias"}
    assert set(batch_stats.keys()) == {"running_mean", "running_var", "num_batches_tracked"}


def test_batch_norm_running_var_is_unbiased():
    """``running_var`` tracks the unbiased variance, matching PyTorch/PyG."""
    key = random.PRNGKey(1)
    x = random.normal(key, (10, 4))

    norm = BatchNorm(4, momentum=0.0, track_running_stats=True, rngs=nnx.Rngs(0))
    norm(x, use_running_average=False)

    n = x.shape[0]
    unbiased = x.var(axis=0) * n / (n - 1)
    assert jnp.allclose(norm.running_var[...], unbiased, atol=1e-5)
    assert jnp.allclose(norm.running_mean[...], x.mean(axis=0), atol=1e-5)


# TODO: HeteroBatchNorm is not implemented in JraphX
# The following test from PyG cannot be converted:
# - test_hetero_batch_norm: Requires HeteroBatchNorm which handles different node types


if __name__ == "__main__":
    test_batch_norm()
    test_batch_norm_single_element()
    test_batch_norm_statistics()
    test_batch_norm_running_stats()
    test_batch_norm_with_batch()
    test_batch_norm_unequal_graphs_are_not_reweighted()
    test_batch_norm_jit_with_batch()
    test_batch_norm_running_stats_are_batch_stats()
    test_batch_norm_running_var_is_unbiased()
    print("All BatchNorm tests passed!")
