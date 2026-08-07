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
    scatter_fallback,
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


def test_scatter_max_preserves_integer_dtype():
    """Empty segments must not leak the integer sentinel nor promote to float."""
    src = jnp.array([[3], [5]], dtype=jnp.int32)
    index = jnp.array([0, 0])

    out = scatter_max(src, index, dim_size=3)

    assert out.dtype == jnp.int32
    assert jnp.array_equal(out, jnp.array([[5], [0], [0]], dtype=jnp.int32))

    out_min = scatter_min(src, index, dim_size=3)
    assert out_min.dtype == jnp.int32
    assert jnp.array_equal(out_min, jnp.array([[3], [0], [0]], dtype=jnp.int32))


def test_scatter_max_exact_for_large_integers():
    """Large int32 values must survive without float32 rounding."""
    src = jnp.array([2**30 + 7, 1], dtype=jnp.int32)
    index = jnp.array([0, 0])

    out = scatter_max(src, index, dim_size=1)

    assert out.dtype == jnp.int32
    assert int(out[0]) == 2**30 + 7


def test_scatter_max_preserves_infinities():
    """A group of only -inf values reports -inf, not the empty-segment fill."""
    src = jnp.array([-jnp.inf, -jnp.inf, 1.0])
    index = jnp.array([0, 0, 1])

    out = scatter_max(src, index, dim_size=3)

    assert bool(jnp.isneginf(out[0]))
    assert float(out[1]) == 1.0
    assert float(out[2]) == 0.0  # genuinely empty segment


def test_scatter_min_preserves_infinities():
    """A group of only +inf values reports +inf, not the empty-segment fill."""
    src = jnp.array([jnp.inf, jnp.inf, 1.0])
    index = jnp.array([0, 0, 1])

    out = scatter_min(src, index, dim_size=3)

    assert bool(jnp.isposinf(out[0]))
    assert float(out[1]) == 1.0
    assert float(out[2]) == 0.0


def test_scatter_max_propagates_nan():
    """NaN in a group must not be silently rewritten to zero."""
    src = jnp.array([jnp.nan, 1.0])
    index = jnp.array([0, 1])

    out_max = scatter_max(src, index, dim_size=2)
    out_min = scatter_min(src, index, dim_size=2)

    assert bool(jnp.isnan(out_max[0]))
    assert bool(jnp.isnan(out_min[0]))


def test_scatter_max_custom_fill_value():
    """``fill_value`` controls what empty segments contain."""
    src = jnp.array([1.0, 2.0])
    index = jnp.array([0, 0])

    out = scatter_max(src, index, dim_size=2, fill_value=-jnp.inf)
    assert float(out[0]) == 2.0
    assert bool(jnp.isneginf(out[1]))

    out = scatter_min(src, index, dim_size=2, fill_value=jnp.inf)
    assert float(out[0]) == 1.0
    assert bool(jnp.isposinf(out[1]))


def test_scatter_logsumexp_empty_segment():
    """Empty segments evaluate to -inf, the identity of log-sum-exp."""
    out = scatter_logsumexp(jnp.array([1.0, 2.0]), jnp.array([0, 0]), dim_size=2)

    assert jnp.allclose(out[0], jnp.log(jnp.exp(1.0) + jnp.exp(2.0)))
    assert bool(jnp.isneginf(out[1]))


def test_scatter_logsumexp_all_neg_inf_group():
    """A group of only -inf values has log-sum-exp -inf, not NaN."""
    out = scatter_logsumexp(jnp.array([-jnp.inf, -jnp.inf]), jnp.array([0, 0]), dim_size=1)

    assert bool(jnp.isneginf(out[0]))


def test_scatter_sum_is_alias_of_add():
    """``reduce='sum'`` is the torch_geometric spelling of ``'add'``."""
    src = jnp.array([[1.0], [2.0], [3.0]])
    index = jnp.array([0, 0, 1])

    assert jnp.array_equal(
        scatter(src, index, dim_size=2, reduce="sum"),
        scatter(src, index, dim_size=2, reduce="add"),
    )
    assert jnp.array_equal(
        scatter_fallback(src, index, dim_size=2, dim=0, reduce="sum"),
        scatter_fallback(src, index, dim_size=2, dim=0, reduce="add"),
    )


def test_scatter_infers_dim_size():
    """Omitting ``dim_size`` infers it from ``index`` in eager execution."""
    src = jnp.array([[1.0], [2.0], [3.0]])
    index = jnp.array([0, 0, 2])

    for reduce_op in ["add", "mean", "max", "min"]:
        assert scatter(src, index, reduce=reduce_op).shape == (3, 1)

    assert jnp.allclose(scatter_add(src, index), jnp.array([[3.0], [0.0], [3.0]]))
    assert scatter_std(src, index).shape == (3, 1)
    assert scatter_logsumexp(src, index).shape == (3, 1)


def test_scatter_std_matches_sample_std():
    """``scatter_std`` applies Bessel's correction by default."""
    src = jnp.array([1.0, 2.0, 3.0, 10.0])
    index = jnp.array([0, 0, 0, 1])

    out = scatter_std(src, index, dim_size=3)
    assert jnp.allclose(out[0], 1.0)  # sample std of [1, 2, 3]
    assert float(out[1]) == 0.0  # single element
    assert float(out[2]) == 0.0  # empty segment

    biased = scatter_std(src, index, dim_size=3, unbiased=False)
    assert jnp.allclose(biased[0], jnp.sqrt(jnp.array(2.0 / 3.0)))


def test_scatter_std_is_stable_for_large_offsets():
    """Centring before squaring avoids the cancellation of E[X^2] - E[X]^2."""
    # Exactly representable in float32, but their squares are not: the
    # E[X^2] - E[X]^2 form loses every significant digit here.
    src = jnp.array([1e5 + 1.0, 1e5 + 2.0, 1e5 + 3.0])
    index = jnp.array([0, 0, 0])

    out = scatter_std(src, index, dim_size=1)
    assert jnp.allclose(out[0], 1.0, atol=1e-3)


def test_scatter_fallback_max_preserves_integer_dtype():
    """The fallback path keeps integer dtypes and zero-fills empty segments."""
    src = jnp.array([[3], [5]], dtype=jnp.int32)
    index = jnp.array([0, 0])

    out = scatter_fallback(src, index, dim_size=3, dim=0, reduce="max")
    assert out.dtype == jnp.int32
    assert jnp.array_equal(out, jnp.array([[5], [0], [0]], dtype=jnp.int32))

    out = scatter_fallback(src, index, dim_size=3, dim=0, reduce="min")
    assert out.dtype == jnp.int32
    assert jnp.array_equal(out, jnp.array([[3], [0], [0]], dtype=jnp.int32))


def test_scatter_fallback_max_preserves_neg_inf():
    """The fallback path does not clobber genuine -inf values."""
    src = jnp.array([-jnp.inf, 1.0])
    index = jnp.array([0, 1])

    out = scatter_fallback(src, index, dim_size=3, dim=0, reduce="max")
    assert bool(jnp.isneginf(out[0]))
    assert float(out[1]) == 1.0
    assert float(out[2]) == 0.0


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


def test_scatter_mean_does_not_saturate_in_low_precision():
    """The mean of a large segment must survive a low-precision input dtype.

    bfloat16 carries 8 mantissa bits, so its consecutive integers stop at 256: 256 + 1
    rounds straight back to 256. Accumulating either the running total or the member
    count in the input dtype therefore freezes both partway through a high-degree node,
    and the quotient comes out wrong by a degree-dependent factor with no warning.
    """
    num_members = 600
    values = (jnp.arange(num_members, dtype=jnp.float32) % 7) + 1.0
    index = jnp.zeros(num_members, dtype=jnp.int32)
    reference = float(values.mean())

    for dtype in (jnp.bfloat16, jnp.float16):
        src = values.astype(dtype).reshape(-1, 1)
        out = scatter_mean(src, index, dim_size=1)

        # The caller's dtype is preserved, so the tolerance is set by its storage
        # precision rather than by the accumulation
        assert out.dtype == dtype
        assert abs(float(out[0, 0]) - reference) < 0.05 * reference

    # float32 is exact to within rounding
    out_f32 = scatter_mean(values.reshape(-1, 1), index, dim_size=1)
    assert out_f32.dtype == jnp.float32
    assert jnp.allclose(out_f32[0, 0], reference, atol=1e-5)


def test_scatter_std_does_not_saturate_in_low_precision():
    """The same widening has to cover the deviation, which divides by a count too."""
    num_members = 600
    values = (jnp.arange(num_members, dtype=jnp.float32) % 7) + 1.0
    index = jnp.zeros(num_members, dtype=jnp.int32)
    reference = float(values.std(ddof=1))

    src = values.astype(jnp.bfloat16).reshape(-1, 1)
    out = scatter_std(src, index, dim_size=1)

    assert out.dtype == jnp.bfloat16
    assert abs(float(out[0, 0]) - reference) < 0.05 * reference


def test_scatter_mean_promotes_integer_input_to_float():
    """An integer input still divides to a float, as `jnp.true_divide` would."""
    src = jnp.array([[1], [3], [5]], dtype=jnp.int32)
    index = jnp.zeros(3, dtype=jnp.int32)
    out = scatter_mean(src, index, dim_size=1)

    assert jnp.issubdtype(out.dtype, jnp.floating)
    assert jnp.allclose(out[0, 0], 3.0)


def test_scatter_logsumexp_does_not_saturate_in_low_precision():
    """logsumexp over a uniform 600-member bfloat16 segment is log(600).

    The exponential sum is accumulated in at least float32; in bfloat16 it
    froze at 256 and the result came out log(256) instead.
    """
    src = jnp.zeros(600, dtype=jnp.bfloat16)
    index = jnp.zeros(600, dtype=jnp.int32)

    out = scatter_logsumexp(src, index, dim_size=1)

    assert out.dtype == jnp.bfloat16
    expected = float(jnp.log(600.0))
    assert abs(float(out[0]) - expected) < 0.02 * expected
