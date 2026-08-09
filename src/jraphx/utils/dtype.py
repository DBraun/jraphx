"""Dtype resolution for user-facing ``dtype`` arguments."""

import jax.numpy as jnp

#: Abstract numpy scalar categories. They satisfy ``issubdtype`` checks against
#: ``jnp.number``, but they cannot be materialized as an array dtype, so
#: accepting them would defer the failure to the first array construction.
_ABSTRACT_CATEGORIES = frozenset(
    {
        "generic",
        "number",
        "inexact",
        "floating",
        "complexfloating",
        "integer",
        "signedinteger",
        "unsignedinteger",
        "flexible",
        "character",
    }
)


def parse_dtype(dtype: str | type | jnp.dtype) -> type:
    """Resolve a dtype spec to the matching jax.numpy scalar type (e.g. ``jnp.float32``).

    Args:
        dtype: A string naming a jax.numpy dtype, with or without a ``jnp.`` /
            ``jax.numpy.`` / ``np.`` / ``numpy.`` prefix (``"float32"``,
            ``"jnp.bfloat16"``, ``"int32"``, ...); an already-resolved scalar
            type such as ``jnp.float32``; or a ``jax.numpy.dtype`` /
            ``numpy.dtype`` object. Non-string dtypes are normalized to the
            jax.numpy scalar type. The Python builtins :obj:`float` and
            :obj:`int` are rejected -- name the width explicitly.

    Returns:
        The matching jax.numpy scalar type, e.g. ``jnp.float32``.

    Raises:
        TypeError: If ``dtype`` is neither a string nor a dtype-like value.
        ValueError: If ``dtype`` does not name a concrete jax.numpy dtype.
            Abstract categories such as ``"floating"`` or ``"integer"`` are
            rejected here rather than failing later at the first array
            construction.
    """
    if isinstance(dtype, str):
        name = dtype.strip()
        for prefix in ("jax.numpy.", "jnp.", "numpy.", "np."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
    elif isinstance(dtype, jnp.dtype):
        name = dtype.name
    elif isinstance(dtype, type):
        name = dtype.__name__
    else:
        raise TypeError(
            f"parse_dtype expects a dtype string or a dtype, got "
            f"{type(dtype).__name__}: {dtype!r}"
        )

    resolved = getattr(jnp, name, None)
    if (
        name not in _ABSTRACT_CATEGORIES
        and isinstance(resolved, type)
        and (jnp.issubdtype(resolved, jnp.number) or jnp.issubdtype(resolved, jnp.bool_))
    ):
        return resolved
    raise ValueError(
        f"{dtype!r} does not name a concrete jax.numpy dtype "
        "(e.g. 'float32', 'jnp.bfloat16', 'int32', 'bool')."
    )
