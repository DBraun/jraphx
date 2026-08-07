"""Tests for the :class:`MessagePassing` propagate contract."""

import jax
import pytest
from flax import nnx
from jax import numpy as jnp

from jraphx.nn.conv.message_passing import MessagePassing


class CountingMP(MessagePassing):
    """Message passing probe that counts how often :meth:`message` runs."""

    def __init__(self, aggr: str = "add", flow: str = "source_to_target"):
        super().__init__(aggr=aggr, flow=flow)
        self.message_calls = 0

    def message(
        self,
        x_j: jax.Array,
        x_i: jax.Array | None = None,
        edge_attr: jax.Array | None = None,
    ) -> jax.Array:
        """Return the source features and record the invocation."""
        self.message_calls += 1
        return x_j


class SumUpdateMP(MessagePassing):
    """Probe whose :meth:`update` combines the aggregate with the target table."""

    def update(self, aggr_out: jax.Array, x: jax.Array | None = None) -> jax.Array:
        """Add the original target features to the aggregated messages."""
        return aggr_out + x


class FusedMP(MessagePassing):
    """Probe that overrides the fused hook; :meth:`message` must never run."""

    def message(
        self,
        x_j: jax.Array,
        x_i: jax.Array | None = None,
        edge_attr: jax.Array | None = None,
    ) -> jax.Array:
        """Fail loudly: the fused path must bypass this method."""
        raise RuntimeError("message() must not be called when message_and_aggregate is overridden")

    def message_and_aggregate(
        self,
        x: jax.Array,
        edge_index: jax.Array,
        edge_attr: jax.Array | None = None,
        dim_size: int | None = None,
    ) -> jax.Array:
        """Return a constant of the aggregated shape."""
        return jnp.full((dim_size, x.shape[-1]), 7.0)


class RawArgsFusedMP(MessagePassing):
    """Probe that checks what the fused hook is handed and what happens after."""

    def __init__(self):
        super().__init__(aggr="add")
        self.x_was_tuple = False
        self.num_source_rows = 0
        self.edge_index_shape = ()

    def message_and_aggregate(
        self,
        x: jax.Array | tuple[jax.Array, jax.Array],
        edge_index: jax.Array,
        edge_attr: jax.Array | None = None,
        dim_size: int | None = None,
    ) -> jax.Array:
        """Scatter the source rows of ``x`` into the target set of ``edge_index``."""
        x_src = x[0] if isinstance(x, tuple) else x
        self.x_was_tuple = isinstance(x, tuple)
        self.num_source_rows = x_src.shape[0]
        self.edge_index_shape = edge_index.shape
        out = jnp.zeros((dim_size, x_src.shape[-1]), dtype=x_src.dtype)
        return out.at[edge_index[1]].add(x_src[edge_index[0]])

    def update(self, aggr_out: jax.Array, x: jax.Array | None = None) -> jax.Array:
        """Offset the fused result to show that :meth:`update` still runs."""
        return aggr_out + 1.0


class CountingMLP(nnx.Module):
    """Linear layer that records how many times it has been applied."""

    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.lin = nnx.Linear(in_features, out_features, rngs=rngs)
        self.calls = 0

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the linear layer and record the invocation."""
        self.calls += 1
        return self.lin(x)


def test_propagate_homogeneous_matches_manual_scatter():
    """A single feature table aggregates into every node of that table."""
    x = jnp.array([[10.0], [20.0], [30.0]])
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    out = CountingMP(aggr="add").propagate(edge_index, x)

    assert out.shape == (3, 1)
    assert jnp.allclose(out, jnp.array([[10.0], [50.0], [0.0]]))


def test_propagate_bipartite_source_to_target():
    """``x = (x_src, x_dst)`` gathers from ``x[0]`` and outputs ``x[1]`` rows."""
    x_src = jnp.array([[10.0], [20.0], [30.0]])
    x_dst = jnp.array([[1.0], [2.0]])
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    out = CountingMP(aggr="add").propagate(edge_index, (x_src, x_dst))

    assert out.shape == (2, 1)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.allclose(out, jnp.array([[10.0], [50.0]]))


def test_propagate_bipartite_unequal_sizes():
    """Bipartite output is sized by the target table, not the source table."""
    x_src = jnp.ones((5, 2))
    x_dst = jnp.full((3, 2), 2.0)
    edge_index = jnp.array([[0, 1, 2, 3, 4], [0, 0, 1, 2, 2]])

    out = CountingMP(aggr="add").propagate(edge_index, (x_src, x_dst))

    assert out.shape == (3, 2)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.allclose(out, jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 2.0]]))


def test_propagate_bipartite_target_to_source():
    """With reversed flow the source table is ``x[1]`` and the output ``x[0]`` rows."""
    x_0 = jnp.array([[1.0], [2.0], [3.0]])
    x_1 = jnp.array([[10.0], [20.0]])
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    conv = CountingMP(aggr="add", flow="target_to_source")
    out = conv.propagate(edge_index, (x_0, x_1))

    assert out.shape == (3, 1)
    assert jnp.allclose(out, jnp.array([[10.0], [20.0], [20.0]]))


def test_propagate_homogeneous_target_to_source():
    """Reversed flow on a single table scatters into the source row of each edge."""
    x = jnp.array([[10.0], [20.0], [30.0]])
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    out = CountingMP(aggr="add", flow="target_to_source").propagate(edge_index, x)

    assert out.shape == (3, 1)
    assert jnp.allclose(out, jnp.array([[10.0], [20.0], [20.0]]))


def test_propagate_bipartite_update_uses_target_table():
    """``update`` receives the target table, so it broadcasts against the aggregate."""
    x_src = jnp.ones((4, 3))
    x_dst = jnp.full((2, 3), 5.0)
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    out = SumUpdateMP(aggr="add").propagate(edge_index, (x_src, x_dst))

    assert out.shape == (2, 3)
    assert jnp.allclose(out, jnp.full((2, 3), 7.0))


def test_propagate_message_runs_once_per_call():
    """``message`` is evaluated exactly once per :meth:`propagate`."""
    conv = CountingMP(aggr="add")
    x = jnp.ones((4, 2))
    edge_index = jnp.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv.propagate(edge_index, x)
    assert conv.message_calls == 1

    conv.propagate(edge_index, x)
    assert conv.message_calls == 2


def test_stateful_message_submodule_runs_once():
    """A submodule invoked from ``message`` executes once per forward pass."""

    class SubmoduleMP(MessagePassing):
        def __init__(self, rngs: nnx.Rngs):
            super().__init__(aggr="add")
            self.nn = CountingMLP(2, 2, rngs=rngs)

        def message(
            self,
            x_j: jax.Array,
            x_i: jax.Array | None = None,
            edge_attr: jax.Array | None = None,
        ) -> jax.Array:
            return self.nn(x_j)

    conv = SubmoduleMP(rngs=nnx.Rngs(0))
    x = jnp.ones((3, 2))
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    conv.propagate(edge_index, x)
    assert conv.nn.calls == 1


def test_base_message_and_aggregate_is_not_dispatched():
    """The base fused hook is never used and raises when called directly."""
    conv = CountingMP(aggr="add")
    x = jnp.ones((3, 2))
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    conv.propagate(edge_index, x)
    assert conv.message_calls == 1

    with pytest.raises(NotImplementedError, match="does not implement a fused"):
        conv.message_and_aggregate(x, edge_index, None, 3)


def test_overridden_message_and_aggregate_is_dispatched():
    """A genuine override takes over the whole message/aggregate step."""
    conv = FusedMP(aggr="add")
    x = jnp.ones((3, 2))
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    out = conv.propagate(edge_index, x)

    assert out.shape == (3, 2)
    assert jnp.allclose(out, 7.0)


def test_fused_hook_receives_raw_arguments_and_feeds_update():
    """The fused hook gets ``x`` and ``edge_index`` untouched, source table first."""
    conv = RawArgsFusedMP()
    x_src = jnp.array([[1.0], [2.0], [3.0]])
    x_dst = jnp.zeros((2, 1))
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    out = conv.propagate(edge_index, (x_src, x_dst))

    assert conv.x_was_tuple
    assert conv.num_source_rows == 3
    assert conv.edge_index_shape == (2, 3)
    # Aggregated [[1], [5]], then offset by the overridden update()
    assert jnp.allclose(out, jnp.array([[2.0], [6.0]]))


def test_propagate_rejects_out_of_range_bipartite_indices():
    """Out-of-range bipartite ids raise instead of silently gathering NaN."""
    conv = CountingMP(aggr="add")
    x_src = jnp.ones((2, 1))
    x_dst = jnp.ones((2, 1))
    edge_index = jnp.array([[0, 5], [0, 1]])

    with pytest.raises(IndexError, match="Source indices"):
        conv.propagate(edge_index, (x_src, x_dst))

    with pytest.raises(IndexError, match="Target indices"):
        conv.propagate(jnp.array([[0, 1], [0, 4]]), (x_src, x_dst))


def test_propagate_rejects_size_disagreeing_with_tables():
    """An explicit ``size`` must match the bipartite feature tables."""
    conv = CountingMP(aggr="add")
    x_src = jnp.ones((4, 1))
    x_dst = jnp.ones((2, 1))
    edge_index = jnp.array([[0, 1], [0, 1]])

    with pytest.raises(ValueError, match="disagrees with the bipartite feature tables"):
        conv.propagate(edge_index, (x_src, x_dst), size=(4, 3))

    out = conv.propagate(edge_index, (x_src, x_dst), size=(4, 2))
    assert out.shape == (2, 1)


def test_propagate_rejects_non_pair_tuple():
    """Only a 2-tuple describes a bipartite pair of node tables."""
    conv = CountingMP(aggr="add")
    x = jnp.ones((2, 1))
    edge_index = jnp.array([[0, 1], [0, 1]])

    with pytest.raises(ValueError, match="must be a 2-tuple"):
        conv.propagate(edge_index, (x, x, x))


def test_propagate_bipartite_under_jit():
    """The bipartite path traces cleanly and keeps its semantics under jit."""
    conv = MessagePassing(aggr="add")
    x_src = jnp.array([[10.0], [20.0], [30.0]])
    x_dst = jnp.zeros((2, 1))
    edge_index = jnp.array([[0, 1, 2], [0, 1, 1]])

    @jax.jit
    def run(x_src: jax.Array, x_dst: jax.Array, edge_index: jax.Array) -> jax.Array:
        return conv.propagate(edge_index, (x_src, x_dst))

    out = run(x_src, x_dst, edge_index)

    assert out.shape == (2, 1)
    assert jnp.allclose(out, jnp.array([[10.0], [50.0]]))
