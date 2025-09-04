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
    assert jnp.allclose(norm.running_mean.value, jnp.zeros(16))
    assert jnp.allclose(norm.running_var.value, jnp.ones(16))
    assert norm.num_batches_tracked.value == 0

    # Process first batch
    _ = norm(x1, use_running_average=False)
    assert norm.num_batches_tracked.value == 1

    # Process second batch
    _ = norm(x2, use_running_average=False)
    assert norm.num_batches_tracked.value == 2

    # Running stats should have been updated
    assert not jnp.allclose(norm.running_mean.value, jnp.zeros(16))
    assert not jnp.allclose(norm.running_var.value, jnp.ones(16))


def test_batch_norm_with_batch():
    """Test BatchNorm with batch assignment vector."""
    key = random.PRNGKey(42)
    x = random.normal(key, (200, 16))
    # Create batch vector: 4 graphs with 50 nodes each
    batch = jnp.repeat(jnp.arange(4), 50)

    norm = BatchNorm(16, track_running_stats=False, rngs=nnx.Rngs(42))
    out = norm(x, batch)

    assert out.shape == (200, 16)

    # Note: per-batch stats are averaged, so individual batch means won't be exactly 0
    # but the overall mean should still be close to 0

    overall_mean = out.mean(axis=0)
    # overall_std = out.std(axis=0)  # Not used in assertions
    assert jnp.allclose(overall_mean, jnp.zeros_like(overall_mean), atol=1e-5)


# TODO: HeteroBatchNorm is not implemented in JraphX
# The following test from PyG cannot be converted:
# - test_hetero_batch_norm: Requires HeteroBatchNorm which handles different node types


if __name__ == "__main__":
    test_batch_norm()
    test_batch_norm_single_element()
    test_batch_norm_statistics()
    test_batch_norm_running_stats()
    test_batch_norm_with_batch()
    print("All BatchNorm tests passed!")
