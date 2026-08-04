"""Tests for :class:`jraphx.data.Data`."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from flax.struct import dataclass

from jraphx.data import Data


@dataclass
class CustomData(Data):
    """A Data subclass built with the documented subclassing recipe."""

    custom_attr: jnp.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        """Delegate to the base class so array fields compare element-wise."""
        return Data.__eq__(self, other)


def _line_graph() -> Data:
    """Build a three-node path graph with two directed edges.

    Returns:
        A Data object with node features, edge indices and edge features.
    """
    return Data(
        x=jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
        edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
        edge_attr=jnp.ones((2, 1), dtype=jnp.float32),
    )


class TestCounts:
    """Shape-derived properties."""

    def test_num_nodes_from_x(self) -> None:
        data = _line_graph()
        assert data.num_nodes == 3
        assert isinstance(data.num_nodes, int)

    def test_num_nodes_from_edge_index_is_python_int(self) -> None:
        data = Data(edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32))
        assert data.num_nodes == 3
        assert isinstance(data.num_nodes, int)

    def test_num_nodes_from_pos(self) -> None:
        data = Data(pos=jnp.zeros((5, 3)))
        assert data.num_nodes == 5

    def test_num_nodes_prefers_pos_over_edge_index(self) -> None:
        """pos counts every node, while edge_index undercounts isolated ones."""
        data = Data(pos=jnp.zeros((5, 3)), edge_index=jnp.array([[0], [1]], dtype=jnp.int32))
        assert data.num_nodes == 5

    def test_num_nodes_empty(self) -> None:
        assert Data().num_nodes == 0

    def test_num_nodes_under_jit_requires_x(self) -> None:
        """The edge_index fallback reads a concrete value, so tracing must fail."""
        data = Data(edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32))
        with pytest.raises(jax.errors.ConcretizationTypeError):
            jax.jit(lambda d: jnp.zeros(d.num_nodes))(data)

    def test_num_edges(self) -> None:
        assert _line_graph().num_edges == 2
        assert Data().num_edges == 0

    def test_num_features(self) -> None:
        data = _line_graph()
        assert data.num_node_features == 2
        assert data.num_edge_features == 1
        assert Data().num_node_features == 0
        assert Data().num_edge_features == 0


class TestKeys:
    """Attribute enumeration."""

    def test_keys_lists_only_populated_fields(self) -> None:
        assert set(_line_graph().keys()) == {"x", "edge_index", "edge_attr"}

    def test_keys_matches_dataclass_fields(self) -> None:
        data = _line_graph()
        field_names = {f.name for f in dataclasses.fields(data)}
        assert set(data.keys()) <= field_names

    def test_contains(self) -> None:
        data = _line_graph()
        assert "x" in data
        assert "pos" not in data


class TestPredicates:
    """Structural predicates return plain Python booleans."""

    def test_is_directed(self) -> None:
        directed = Data(x=jnp.zeros((2, 1)), edge_index=jnp.array([[0], [1]], dtype=jnp.int32))
        undirected = Data(
            x=jnp.zeros((2, 1)), edge_index=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        )
        assert directed.is_directed is True
        assert undirected.is_directed is False

    def test_is_directed_without_edges(self) -> None:
        assert Data(x=jnp.zeros((2, 1))).is_directed is False

    def test_has_self_loops(self) -> None:
        with_loop = Data(x=jnp.zeros((2, 1)), edge_index=jnp.array([[0], [0]], dtype=jnp.int32))
        without_loop = Data(x=jnp.zeros((2, 1)), edge_index=jnp.array([[0], [1]], dtype=jnp.int32))
        assert with_loop.has_self_loops() is True
        assert without_loop.has_self_loops() is False

    def test_has_isolated_nodes(self) -> None:
        isolated = Data(x=jnp.zeros((3, 1)), edge_index=jnp.array([[0], [1]], dtype=jnp.int32))
        connected = Data(
            x=jnp.zeros((2, 1)), edge_index=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        )
        assert isolated.has_isolated_nodes() is True
        assert connected.has_isolated_nodes() is False


class TestPyTree:
    """Data must be usable directly inside JAX transformations."""

    def test_leaves_are_arrays(self) -> None:
        leaves = jax.tree_util.tree_leaves(_line_graph())
        assert len(leaves) == 3
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_jit(self) -> None:
        data = _line_graph()
        assert jax.jit(lambda d: d.x.sum())(data) == data.x.sum()

    def test_tree_map_returns_data(self) -> None:
        doubled = jax.tree.map(lambda a: a * 2, _line_graph())
        assert isinstance(doubled, Data)
        assert jnp.array_equal(doubled.x, _line_graph().x * 2)


class TestRepr:
    """String representation."""

    def test_repr_reports_shapes(self) -> None:
        text = repr(_line_graph())
        assert text.startswith("Data(")
        assert "x=[3, 2]" in text
        assert "edge_index=[2, 2]" in text
        assert "pos" not in text


class TestEquality:
    """Equality compares array fields element-wise instead of truth-testing them."""

    def test_equal_contents_compare_equal(self) -> None:
        assert _line_graph() == _line_graph()

    def test_differing_contents_compare_unequal(self) -> None:
        assert _line_graph() != _line_graph().replace(x=jnp.zeros((3, 2)))

    def test_differing_shapes_compare_unequal(self) -> None:
        assert Data(x=jnp.zeros((3, 2))) != Data(x=jnp.zeros((2, 2)))

    def test_unset_field_differs_from_set_field(self) -> None:
        assert Data(x=jnp.zeros((3, 2))) != Data()

    def test_other_types_compare_unequal(self) -> None:
        assert _line_graph() != 5

    def test_membership_uses_equality(self) -> None:
        assert _line_graph() in [_line_graph()]

    def test_subclass_delegating_eq_compares_element_wise(self) -> None:
        """The documented subclassing recipe keeps == working on array fields."""
        assert CustomData(x=jnp.zeros((3, 2))) == CustomData(x=jnp.zeros((3, 2)))
        assert CustomData(custom_attr=jnp.zeros((2,))) != CustomData(custom_attr=jnp.ones((2,)))
        assert CustomData(x=jnp.zeros((3, 2))) != Data(x=jnp.zeros((3, 2)))


class TestImmutability:
    """flax.struct dataclasses are frozen."""

    def test_assignment_is_rejected(self) -> None:
        data = _line_graph()
        with pytest.raises(dataclasses.FrozenInstanceError):
            data.x = jnp.zeros((3, 2))

    def test_replace_creates_new_instance(self) -> None:
        data = _line_graph()
        updated = data.replace(x=jnp.ones((3, 2)))
        assert jnp.array_equal(updated.x, jnp.ones((3, 2)))
        assert jnp.array_equal(data.x, _line_graph().x)
