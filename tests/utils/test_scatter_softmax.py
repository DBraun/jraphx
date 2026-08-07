"""Tests for the grouped softmax utilities used by attention layers."""

import jax
import jax.numpy as jnp
import pytest

from jraphx.utils.scatter_softmax import (
    masked_scatter_softmax,
    scatter_log_softmax,
    scatter_softmax,
)


def test_scatter_softmax_matches_dense_softmax():
    """Each group reproduces a dense softmax over its own values."""
    src = jnp.array([1.0, 2.0, 3.0, 1.5])
    index = jnp.array([0, 0, 1, 1])

    out = scatter_softmax(src, index, dim_size=2)

    expected = jnp.concatenate(
        [jax.nn.softmax(jnp.array([1.0, 2.0])), jax.nn.softmax(jnp.array([3.0, 1.5]))]
    )
    assert jnp.allclose(out, expected)


def test_scatter_softmax_groups_sum_to_one():
    """Softmax weights are normalized within each group."""
    src = jnp.array([[0.5], [1.5], [-2.0], [0.0]])
    index = jnp.array([0, 0, 0, 1])

    out = scatter_softmax(src, index, dim_size=2)

    assert jnp.allclose(out[:3].sum(), 1.0)
    assert jnp.allclose(out[3].sum(), 1.0)


def test_scatter_softmax_fully_masked_group_is_zero():
    """A group whose entries are all -inf yields zeros rather than NaN."""
    src = jnp.array([-jnp.inf, -jnp.inf, 1.0])
    index = jnp.array([0, 0, 1])

    out = scatter_softmax(src, index, dim_size=2)

    assert not bool(jnp.any(jnp.isnan(out)))
    assert jnp.allclose(out[:2], jnp.zeros(2))
    assert jnp.allclose(out[2], 1.0)


def test_scatter_softmax_temperature():
    """Temperature scales the logits before normalization."""
    src = jnp.array([1.0, 2.0])
    index = jnp.array([0, 0])

    out = scatter_softmax(src, index, dim_size=1, temperature=2.0)

    assert jnp.allclose(out, jax.nn.softmax(src / 2.0))


def test_scatter_softmax_infers_dim_size():
    """Omitting ``dim_size`` infers it from ``index`` in eager execution."""
    src = jnp.array([1.0, 2.0, 3.0])
    index = jnp.array([0, 0, 1])

    assert scatter_softmax(src, index).shape == (3,)
    assert scatter_log_softmax(src, index).shape == (3,)


def test_scatter_softmax_is_jittable_with_static_dim_size():
    """A static ``dim_size`` makes the op traceable."""
    fn = jax.jit(lambda s, i: scatter_softmax(s, i, dim_size=2))

    out = fn(jnp.array([1.0, 2.0, 3.0]), jnp.array([0, 0, 1]))

    assert jnp.allclose(out[:2].sum(), 1.0)


def test_scatter_log_softmax_matches_dense():
    """Log-softmax matches a dense reference per group."""
    src = jnp.array([1.0, 2.0, 3.0, 1.5])
    index = jnp.array([0, 0, 1, 1])

    out = scatter_log_softmax(src, index, dim_size=2)

    expected = jnp.concatenate(
        [
            jax.nn.log_softmax(jnp.array([1.0, 2.0])),
            jax.nn.log_softmax(jnp.array([3.0, 1.5])),
        ]
    )
    assert jnp.allclose(out, expected, atol=1e-6)


def test_scatter_log_softmax_exponentiates_to_softmax():
    """exp(log_softmax) reproduces the softmax weights."""
    src = jnp.array([[0.3], [1.2], [-0.7]])
    index = jnp.array([0, 1, 1])

    log_out = scatter_log_softmax(src, index, dim_size=2)
    out = scatter_softmax(src, index, dim_size=2)

    assert jnp.allclose(jnp.exp(log_out), out, atol=1e-6)


def test_scatter_log_softmax_fully_masked_group_is_neg_inf():
    """A group whose entries are all -inf yields -inf rather than NaN."""
    src = jnp.array([-jnp.inf, -jnp.inf, 1.0])
    index = jnp.array([0, 0, 1])

    log_out = scatter_log_softmax(src, index, dim_size=2)

    assert not bool(jnp.any(jnp.isnan(log_out)))
    assert bool(jnp.all(jnp.isneginf(log_out[:2])))
    assert jnp.allclose(log_out[2], 0.0)

    # Exponentiating reproduces the zeros that ``scatter_softmax`` returns.
    assert jnp.allclose(jnp.exp(log_out), scatter_softmax(src, index, dim_size=2))


def test_scatter_log_softmax_partially_masked_group():
    """Masked entries drop out without perturbing their group's finite entries."""
    src = jnp.array([1.0, -jnp.inf, 3.0])
    index = jnp.array([0, 0, 0])

    log_out = scatter_log_softmax(src, index, dim_size=1)

    expected = jax.nn.log_softmax(jnp.array([1.0, 3.0]))
    assert bool(jnp.isneginf(log_out[1]))
    assert jnp.allclose(jnp.array([log_out[0], log_out[2]]), expected, atol=1e-6)


def test_masked_scatter_softmax_excludes_masked_entries():
    """Masked entries neither contribute mass nor appear in the output."""
    src = jnp.array([1.0, 2.0, 3.0])
    index = jnp.array([0, 0, 0])
    mask = jnp.array([True, False, True])

    out = masked_scatter_softmax(src, index, mask=mask, dim_size=1)

    expected = jax.nn.softmax(jnp.array([1.0, 3.0]))
    assert float(out[1]) == 0.0
    assert jnp.allclose(jnp.array([out[0], out[2]]), expected)


def test_masked_scatter_softmax_all_masked_group():
    """A fully masked group produces zeros, not NaN."""
    src = jnp.array([1.0, 2.0])
    index = jnp.array([0, 0])
    mask = jnp.array([False, False])

    out = masked_scatter_softmax(src, index, mask=mask, dim_size=1)

    assert jnp.allclose(out, jnp.zeros(2))


def test_scatter_softmax_rejects_unsupported_dim():
    """Only scattering along the leading dimension is implemented."""
    with pytest.raises(NotImplementedError):
        scatter_softmax(jnp.array([1.0, 2.0]), jnp.array([0, 0]), dim_size=1, dim=2)


def test_scatter_softmax_does_not_saturate_in_low_precision():
    """A 600-member bfloat16 group still normalizes to 1.

    bfloat16 integers stop at 256, so an exp-sum accumulated in the input
    dtype froze there and the group's weights summed to well over 2.
    """
    src = jnp.zeros(600, dtype=jnp.bfloat16)
    index = jnp.zeros(600, dtype=jnp.int32)

    weights = scatter_softmax(src, index, dim_size=1)

    assert weights.dtype == jnp.bfloat16
    total = float(jnp.sum(weights.astype(jnp.float32)))
    assert abs(total - 1.0) < 1e-2


def test_scatter_log_softmax_does_not_saturate_in_low_precision():
    """log-softmax of a uniform 600-member bfloat16 group is -log(600)."""
    src = jnp.zeros(600, dtype=jnp.bfloat16)
    index = jnp.zeros(600, dtype=jnp.int32)

    out = scatter_log_softmax(src, index, dim_size=1)

    assert out.dtype == jnp.bfloat16
    expected = -jnp.log(600.0)
    assert abs(float(out[0]) - float(expected)) < 0.02 * abs(float(expected))
