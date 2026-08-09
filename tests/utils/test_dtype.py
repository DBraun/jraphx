"""Tests for :func:`jraphx.utils.parse_dtype`."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jraphx.nn.norm import LayerNorm
from jraphx.utils import degree, parse_dtype


@pytest.mark.parametrize(
    "spec",
    ["float32", "jnp.float32", "jax.numpy.float32", "np.float32", "numpy.float32"],
)
def test_parse_dtype_strips_prefixes(spec):
    assert parse_dtype(spec) is jnp.float32


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("bfloat16", jnp.bfloat16),
        ("float16", jnp.float16),
        ("int32", jnp.int32),
        ("bool", jnp.bool),
        (jnp.float32, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16),
        (np.dtype("float32"), jnp.float32),
        (jnp.dtype("int32"), jnp.int32),
    ],
)
def test_parse_dtype_resolves(spec, expected):
    resolved = parse_dtype(spec)
    assert resolved is expected
    # The result is usable as an array dtype
    assert jnp.zeros(2, dtype=resolved).dtype == jnp.dtype(expected)


@pytest.mark.parametrize(
    "spec", ["floating", "number", "integer", "inexact", "signedinteger", np.floating]
)
def test_parse_dtype_rejects_abstract_categories(spec):
    """Abstract categories pass ``issubdtype`` checks but cannot build arrays.

    Rejecting them here keeps the failure at the call site instead of at the
    first array construction.
    """
    with pytest.raises(ValueError):
        parse_dtype(spec)


@pytest.mark.parametrize("spec", ["float63", "spam", "", float, int])
def test_parse_dtype_rejects_non_dtypes(spec):
    with pytest.raises(ValueError):
        parse_dtype(spec)


def test_parse_dtype_rejects_non_dtype_objects():
    with pytest.raises(TypeError):
        parse_dtype(3.14)


def test_degree_accepts_string_dtype():
    row = jnp.array([0, 1, 0, 2, 0])
    deg = degree(row, dtype="jnp.int32")
    assert deg.dtype == jnp.int32
    assert deg.tolist() == [3, 1, 1]


def test_layer_norm_accepts_string_dtypes():
    norm = LayerNorm(4, dtype="bfloat16", param_dtype="float32", rngs=nnx.Rngs(0))
    assert norm.dtype is jnp.bfloat16
    assert norm.weight[...].dtype == jnp.float32

    out = norm(jnp.ones((3, 4)))
    assert out.dtype == jnp.bfloat16
