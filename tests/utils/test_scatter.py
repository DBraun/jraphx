"""Tests for scatter operations converted from PyTorch Geometric to JraphX.

This module tests the scatter functionality in JraphX, which provides
optimized scatter operations using JAX's built-in operations.
"""

import pytest
from jax import numpy as jnp
from jax import random

from jraphx.utils.scatter import (
    scatter,
    scatter_add,
    scatter_logsumexp,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_std,
)


def test_scatter_validate():
    """Test scatter input validation."""
    key = random.key(0)
    src = random.normal(key, (100, 32))
    index = random.randint(random.key(1), (100,), 0, 10)

    # Test multi-dimensional index (should raise ValueError)
    with pytest.raises(ValueError, match="must be one-dimensional"):
        scatter(src, index.reshape(-1, 1))

    # Test invalid dimension - JraphX only supports dim=0 and dim=-2
    # Note: PyG supports dim=2, but JraphX is more restrictive
    with pytest.raises(NotImplementedError):
        scatter(src, index, dim=2)

    # Test invalid reduce argument - 'std' is now supported in JraphX, so let's test unsupported one
    with pytest.raises(ValueError, match="Unknown reduce operation"):
        scatter(src, index, reduce="invalid_reduce")


def test_scatter_basic():
    """Test basic scatter operations with different reductions."""
    key = random.key(42)
    src = random.normal(key, (100, 16))
    index = random.randint(random.key(43), (100,), 0, 8)

    for reduce_op in ["add", "mean", "max", "min"]:
        out = scatter(src, index, dim=0, reduce=reduce_op, dim_size=8)
        assert out.shape == (8, 16)
        assert not jnp.any(jnp.isnan(out))

    # Test dim=-2 (should work the same as dim=0 for 2D tensors)
    out = scatter(src, index, dim=-2, reduce="add", dim_size=8)
    assert out.shape == (8, 16)


def test_scatter_specific_functions():
    """Test specific scatter functions directly."""
    key = random.key(123)
    src = random.normal(key, (50, 10))
    index = random.randint(random.key(124), (50,), 0, 5)

    # Test each specific function
    out_add = scatter_add(src, index, dim_size=5)
    out_mean = scatter_mean(src, index, dim_size=5)
    out_max = scatter_max(src, index, dim_size=5)
    out_min = scatter_min(src, index, dim_size=5)

    assert out_add.shape == (5, 10)
    assert out_mean.shape == (5, 10)
    assert out_max.shape == (5, 10)
    assert out_min.shape == (5, 10)


def test_scatter_advanced_functions():
    """Test advanced scatter functions (std, logsumexp)."""
    key = random.key(456)
    src = random.normal(key, (30, 8))
    index = random.randint(random.key(457), (30,), 0, 6)

    # Test scatter_std
    out_std = scatter_std(src, index, dim_size=6)
    assert out_std.shape == (6, 8)
    assert jnp.all(out_std >= 0)  # Standard deviation is always non-negative

    # Test scatter_logsumexp
    out_logsumexp = scatter_logsumexp(src, index, dim_size=6)
    assert out_logsumexp.shape == (6, 8)
    # Logsumexp should be numerically stable (no inf/nan for reasonable inputs)
    assert jnp.all(jnp.isfinite(out_logsumexp))


def test_scatter_empty_index():
    """Test scatter with empty index."""
    src = jnp.array([]).reshape(0, 5)
    index = jnp.array([], dtype=jnp.int32)

    out = scatter(src, index, dim_size=3, reduce="add")
    expected = jnp.zeros((3, 5))
    assert jnp.allclose(out, expected)


def test_scatter_consistency():
    """Test that scatter produces consistent results."""
    key = random.key(789)
    src = random.normal(key, (20, 4))
    index = jnp.array([0, 0, 1, 1, 2, 2] * 3 + [0, 1])  # 20 elements

    # Test that generic scatter matches specific functions
    out_generic_add = scatter(src, index, dim_size=3, reduce="add")
    out_specific_add = scatter_add(src, index, dim_size=3)
    assert jnp.allclose(out_generic_add, out_specific_add)

    out_generic_mean = scatter(src, index, dim_size=3, reduce="mean")
    out_specific_mean = scatter_mean(src, index, dim_size=3)
    assert jnp.allclose(out_generic_mean, out_specific_mean)


def test_scatter_single_element():
    """Test scatter with single element per group."""
    src = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = jnp.array([0, 1, 2])

    # All reductions should give the same result for single elements
    out_add = scatter(src, index, dim_size=3, reduce="add")
    out_mean = scatter(src, index, dim_size=3, reduce="mean")
    out_max = scatter(src, index, dim_size=3, reduce="max")
    out_min = scatter(src, index, dim_size=3, reduce="min")

    assert jnp.allclose(out_add, src)
    assert jnp.allclose(out_mean, src)
    assert jnp.allclose(out_max, src)
    assert jnp.allclose(out_min, src)


# TODO: The following features from PyG are not supported in JraphX:
# - JIT scripting (torch.jit.script) - JAX has its own JIT compilation
# - Device-specific operations (CUDA, MPS) - JAX handles device placement differently
# - Gradient computation tests - JAX uses different autodiff system
# - torch_scatter package comparison - JraphX uses JAX's built-in operations
# - Benchmarking framework - would need JAX-specific benchmarking tools
# - "any" reduction - not commonly used in GNN context, not implemented
# - group_argsort, group_cat, scatter_argmax - specialized functions not yet implemented

# The core scatter functionality (add, mean, max, min) is fully supported
# and optimized using JAX's segment operations for better performance.


if __name__ == "__main__":
    # Run basic tests
    test_scatter_validate()
    test_scatter_basic()
    test_scatter_specific_functions()
    test_scatter_advanced_functions()
    test_scatter_empty_index()
    test_scatter_consistency()
    test_scatter_single_element()
    print("All scatter tests passed!")
