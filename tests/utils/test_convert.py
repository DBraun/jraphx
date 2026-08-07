"""Tests for the graph conversion utilities.

Covers :func:`to_undirected`, :func:`to_dense_adj` and :func:`to_edge_index`,
whose semantics mirror the corresponding :obj:`torch_geometric` helpers.
"""

import jax.numpy as jnp
import pytest

from jraphx.utils.convert import to_dense_adj, to_edge_index, to_undirected
from jraphx.utils.degree import degree


def test_to_undirected_already_undirected():
    """An already-undirected graph round-trips without doubling its edges."""
    edge_index = jnp.array([[0, 1], [1, 0]])

    out_index, out_attr = to_undirected(edge_index)

    assert jnp.array_equal(out_index, jnp.array([[0, 1], [1, 0]]))
    assert out_attr is None


def test_to_undirected_adds_reverse_edges():
    """A directed edge gains its reverse and nothing else."""
    edge_index = jnp.array([[0, 1], [1, 2]])

    out_index, _ = to_undirected(edge_index)

    assert jnp.array_equal(out_index, jnp.array([[0, 1, 1, 2], [1, 0, 2, 1]]))


def test_to_undirected_merges_duplicate_attributes():
    """``reduce`` is honored when merging duplicated edges."""
    edge_index = jnp.array([[0, 0], [1, 1]])
    edge_attr = jnp.array([1.0, 2.0])

    out_index, out_attr = to_undirected(edge_index, edge_attr, reduce="mean")
    assert jnp.array_equal(out_index, jnp.array([[0, 1], [1, 0]]))
    assert jnp.allclose(out_attr, jnp.array([1.5, 1.5]))

    out_index, out_attr = to_undirected(edge_index, edge_attr, reduce="add")
    assert jnp.allclose(out_attr, jnp.array([3.0, 3.0]))


def test_to_undirected_degree_is_not_doubled():
    """Degrees computed on the output match the symmetric graph."""
    edge_index = jnp.array([[0, 1], [1, 0]])

    out_index, _ = to_undirected(edge_index, num_nodes=2)

    assert jnp.allclose(degree(out_index[1], num_nodes=2), jnp.array([1.0, 1.0]))


def test_to_undirected_honors_num_nodes():
    """Isolated nodes are allowed via an explicit ``num_nodes``."""
    edge_index = jnp.array([[0], [1]])

    out_index, _ = to_undirected(edge_index, num_nodes=4)

    assert jnp.array_equal(out_index, jnp.array([[0, 1], [1, 0]]))
    assert jnp.allclose(degree(out_index[1], num_nodes=4), jnp.array([1.0, 1.0, 0.0, 0.0]))


def test_to_undirected_rejects_out_of_range_num_nodes():
    """A too-small ``num_nodes`` is an error rather than silent corruption."""
    edge_index = jnp.array([[0, 4], [1, 2]])

    with pytest.raises(ValueError, match="out of range"):
        to_undirected(edge_index, num_nodes=3)


def test_to_undirected_empty():
    """An empty graph converts to an empty graph."""
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)

    out_index, out_attr = to_undirected(edge_index)

    assert out_index.shape == (2, 0)
    assert out_attr is None


def test_to_dense_adj_binary():
    """Without attributes the dense matrix counts parallel edges."""
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

    adj = to_dense_adj(edge_index)

    expected = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert jnp.allclose(adj, expected)


def test_to_dense_adj_accumulates_duplicate_edges():
    """Duplicate edges are summed, not overwritten."""
    edge_index = jnp.array([[0, 0], [1, 1]])

    assert jnp.allclose(to_dense_adj(edge_index)[0, 1], 2.0)

    edge_attr = jnp.array([1.0, 2.0])
    assert jnp.allclose(to_dense_adj(edge_index, edge_attr)[0, 1], 3.0)

    edge_attr = jnp.array([[1.0, 0.5], [2.0, 0.25]])
    adj = to_dense_adj(edge_index, edge_attr)
    assert adj.shape == (2, 2, 2)
    assert jnp.allclose(adj[0, 1], jnp.array([3.0, 0.75]))


def test_to_dense_adj_max_num_nodes():
    """``max_num_nodes`` pads the output for isolated nodes."""
    edge_index = jnp.array([[0], [1]])

    adj = to_dense_adj(edge_index, max_num_nodes=4)

    assert adj.shape == (4, 4)
    assert jnp.allclose(adj.sum(), 1.0)


def test_to_edge_index_keeps_unit_weights():
    """Weights that all happen to be 1 are still returned."""
    adj = jnp.eye(3)

    edge_index, edge_attr = to_edge_index(adj)

    assert jnp.array_equal(edge_index, jnp.array([[0, 1, 2], [0, 1, 2]]))
    assert edge_attr is not None
    assert jnp.allclose(edge_attr, jnp.ones(3))


def test_to_edge_index_round_trip():
    """``to_dense_adj`` and ``to_edge_index`` are inverses for simple graphs."""
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])
    edge_attr = jnp.array([1.0, 2.0, 3.0])

    adj = to_dense_adj(edge_index, edge_attr)
    out_index, out_attr = to_edge_index(adj)

    assert jnp.array_equal(out_index, jnp.array([[0, 1, 2], [1, 2, 0]]))
    assert jnp.allclose(out_attr, jnp.array([1.0, 2.0, 3.0]))


def test_to_edge_index_multi_feature():
    """Feature tensors keep every edge with at least one non-zero feature."""
    edge_index = jnp.array([[0, 1], [1, 0]])
    edge_attr = jnp.array([[1.0, 0.0], [0.0, 2.0]])

    adj = to_dense_adj(edge_index, edge_attr)
    out_index, out_attr = to_edge_index(adj)

    assert jnp.array_equal(out_index, jnp.array([[0, 1], [1, 0]]))
    assert jnp.allclose(out_attr, edge_attr)
