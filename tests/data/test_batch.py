"""Tests for :class:`jraphx.data.Batch`."""

from typing import ClassVar, Optional

import jax
import jax.numpy as jnp
import pytest
from flax.struct import dataclass

from jraphx.data import Batch, Data


@dataclass
class FaceData(Data):
    """A Data subclass carrying triangle connectivity and per-face attributes."""

    face: jnp.ndarray | None = None
    normal: jnp.ndarray | None = None
    label: jnp.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        """Delegate to the base class so array fields compare element-wise."""
        return Data.__eq__(self, other)


@dataclass
class FaceBatch(Batch):
    """A Batch subclass declaring how the extra FaceData fields are collated."""

    face: jnp.ndarray | None = None
    normal: jnp.ndarray | None = None
    label: jnp.ndarray | None = None

    NODE_INDEX_FIELDS: ClassVar[set[str]] = {"face"}
    ELEMENT_LEVEL_FIELDS: ClassVar[set[str]] = {"normal"}
    GRAPH_LEVEL_FIELDS: ClassVar[set[str]] = {"label"}
    _DATA_CLASS: ClassVar[type | None] = FaceData

    def __eq__(self, other: object) -> bool:
        """Delegate to the base class so array fields compare element-wise."""
        return Batch.__eq__(self, other)


def _graph(num_nodes: int, edges: list[list[int]]) -> Data:
    """Build a small graph with one feature column per node.

    Args:
        num_nodes: Number of nodes in the graph.
        edges: Two rows of equal length holding source and target indices.

    Returns:
        A Data object with ``x`` and ``edge_index`` populated.
    """
    return Data(
        x=jnp.arange(num_nodes, dtype=jnp.float32).reshape(num_nodes, 1),
        edge_index=jnp.array(edges, dtype=jnp.int32).reshape(2, -1),
    )


class TestFromDataList:
    """Collation of a list of graphs into one disconnected graph."""

    def test_offsets_and_batch_vector(self) -> None:
        batch = Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])])
        assert batch.x.shape == (5, 1)
        assert jnp.array_equal(batch.edge_index, jnp.array([[0, 1, 3], [1, 2, 4]]))
        assert jnp.array_equal(batch.batch, jnp.array([0, 0, 0, 1, 1]))
        assert jnp.array_equal(batch.ptr, jnp.array([0, 3, 5]))
        assert batch.num_graphs == 2

    def test_empty_list(self) -> None:
        batch = Batch.from_data_list([])
        assert batch.x is None
        assert batch.num_graphs == 1

    def test_graph_without_edges_still_batches(self) -> None:
        """edge_index is exempt from the all-or-none rule: an edgeless graph is valid."""
        batch = Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), Data(x=jnp.zeros((2, 1)))])
        assert batch.x.shape == (5, 1)
        assert batch.batch.shape == (5,)
        assert batch.edge_index.shape == (2, 2)

    def test_edgeless_graph_batches_next_to_edge_attr(self) -> None:
        """edge_attr aligns with the edge axis, so an edgeless graph may omit it."""
        with_edges = Data(
            x=jnp.zeros((3, 1)),
            edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            edge_attr=jnp.ones((2, 1)),
        )
        edgeless = Data(x=jnp.zeros((2, 1)))
        batch = Batch.from_data_list([with_edges, edgeless])
        assert batch.edge_index.shape == (2, 2)
        assert batch.edge_attr.shape == (2, 1)
        assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]

    def test_explicitly_empty_edge_list_batches_next_to_edge_attr(self) -> None:
        with_edges = Data(
            x=jnp.zeros((3, 1)),
            edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            edge_attr=jnp.ones((2, 1)),
        )
        edgeless = Data(x=jnp.zeros((2, 1)), edge_index=jnp.zeros((2, 0), dtype=jnp.int32))
        batch = Batch.from_data_list([with_edges, edgeless])
        assert batch.edge_attr.shape == (2, 1)
        assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]

    def test_edge_attr_missing_on_a_graph_with_edges_is_rejected(self) -> None:
        """Omitting edge_attr while contributing edges would misalign the edge axis."""
        with_attr = Data(
            x=jnp.zeros((3, 1)),
            edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            edge_attr=jnp.ones((2, 1)),
        )
        without_attr = Data(x=jnp.zeros((2, 1)), edge_index=jnp.array([[0], [1]], dtype=jnp.int32))
        with pytest.raises(RuntimeError, match="'edge_attr' has 0 rows on graph 1"):
            Batch.from_data_list([with_attr, without_attr])

    def test_edge_attr_row_count_must_match_edge_count(self) -> None:
        mismatched = Data(
            x=jnp.zeros((3, 1)),
            edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            edge_attr=jnp.ones((3, 1)),
        )
        with pytest.raises(RuntimeError, match="'edge_attr' has 3 rows on graph 0"):
            Batch.from_data_list([mismatched])

    def test_partially_present_attribute_is_rejected(self) -> None:
        """x on only one of two graphs would misalign with the batch vector."""
        with pytest.raises(RuntimeError, match="present on 1 of 2 graphs"):
            Batch.from_data_list(
                [Data(x=jnp.zeros((3, 4))), Data(edge_index=jnp.array([[0], [1]]))]
            )

    def test_partially_present_pos_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="'pos'"):
            Batch.from_data_list(
                [
                    Data(x=jnp.zeros((3, 4)), pos=jnp.zeros((3, 2))),
                    Data(x=jnp.zeros((2, 4))),
                ]
            )

    def test_scalar_y_is_stacked(self) -> None:
        batch = Batch.from_data_list(
            [
                Data(x=jnp.zeros((3, 1)), y=jnp.array(0.0)),
                Data(x=jnp.zeros((2, 1)), y=jnp.array(1.0)),
            ]
        )
        assert batch.y.shape == (2,)

    def test_node_level_y_is_concatenated(self) -> None:
        batch = Batch.from_data_list(
            [
                Data(x=jnp.zeros((3, 1)), y=jnp.zeros((3,))),
                Data(x=jnp.zeros((2, 1)), y=jnp.ones((2,))),
            ]
        )
        assert batch.y.shape == (5,)

    def test_single_node_graphs_keep_node_level_y_on_axis_zero(self) -> None:
        """A length-1 label is concatenated, matching torch_geometric collation."""
        batch = Batch.from_data_list(
            [
                Data(x=jnp.zeros((1, 1)), y=jnp.array([3.0])),
                Data(x=jnp.zeros((1, 1)), y=jnp.array([4.0])),
            ]
        )
        assert batch.y.shape == (2,)
        assert jnp.array_equal(batch.y, jnp.array([3.0, 4.0]))

    def test_existing_batch_and_ptr_are_recomputed(self) -> None:
        first = Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])])
        second = Batch.from_data_list([first])
        assert jnp.array_equal(second.batch, jnp.zeros(5, dtype=jnp.int32))
        assert jnp.array_equal(second.ptr, jnp.array([0, 5]))


class TestNumGraphs:
    """The graph count must come from the lossless source."""

    def test_trailing_empty_graph_is_counted(self) -> None:
        batch = Batch.from_data_list(
            [
                _graph(3, [[0, 1], [1, 2]]),
                Data(x=jnp.zeros((0, 1)), edge_index=jnp.zeros((2, 0), dtype=jnp.int32)),
            ]
        )
        assert jnp.array_equal(batch.ptr, jnp.array([0, 3, 3]))
        assert batch.num_graphs == 2

    def test_trailing_empty_graph_survives_round_trip(self) -> None:
        batch = Batch.from_data_list(
            [
                _graph(3, [[0, 1], [1, 2]]),
                Data(x=jnp.zeros((0, 1)), edge_index=jnp.zeros((2, 0), dtype=jnp.int32)),
            ]
        )
        graphs = batch.to_data_list()
        assert len(graphs) == 2
        assert graphs[0].num_nodes == 3
        assert graphs[1].num_nodes == 0
        assert graphs[1].edge_index.shape == (2, 0)

    def test_falls_back_to_batch_vector(self) -> None:
        batch = Batch(x=jnp.zeros((4, 1)), batch=jnp.array([0, 0, 1, 1]))
        assert batch.num_graphs == 2
        assert isinstance(batch.num_graphs, int)

    def test_defaults_to_one(self) -> None:
        assert Batch(x=jnp.zeros((2, 1))).num_graphs == 1


class TestToDataList:
    """Splitting a batch back into individual graphs."""

    def test_round_trip(self) -> None:
        originals = [_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])]
        graphs = Batch.from_data_list(originals).to_data_list()
        assert len(graphs) == 2
        for original, recovered in zip(originals, graphs, strict=True):
            assert jnp.array_equal(original.x, recovered.x)
            assert jnp.array_equal(original.edge_index, recovered.edge_index)

    def test_edge_attr_round_trip(self) -> None:
        first = Data(
            x=jnp.zeros((3, 1)),
            edge_index=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            edge_attr=jnp.array([[1.0], [2.0]]),
        )
        second = Data(
            x=jnp.zeros((2, 1)),
            edge_index=jnp.array([[0], [1]], dtype=jnp.int32),
            edge_attr=jnp.array([[3.0]]),
        )
        graphs = Batch.from_data_list([first, second]).to_data_list()
        assert jnp.array_equal(graphs[0].edge_attr, first.edge_attr)
        assert jnp.array_equal(graphs[1].edge_attr, second.edge_attr)

    def test_interleaved_batch_vector_remaps_edges(self) -> None:
        """Node ids are remapped by rank, not by subtracting the minimum id."""
        batch = Batch(
            x=jnp.arange(4, dtype=jnp.float32).reshape(4, 1),
            edge_index=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
            batch=jnp.array([0, 1, 0, 1]),
        )
        graphs = batch.to_data_list()
        assert len(graphs) == 2
        for graph in graphs:
            assert graph.num_nodes == 2
            assert jnp.array_equal(graph.edge_index, jnp.array([[0], [1]]))
        assert jnp.array_equal(graphs[0].x, jnp.array([[0.0], [2.0]]))
        assert jnp.array_equal(graphs[1].x, jnp.array([[1.0], [3.0]]))

    def test_edgeless_graph_keeps_empty_edge_index(self) -> None:
        batch = Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), Data(x=jnp.zeros((2, 1)))])
        graphs = batch.to_data_list()
        assert graphs[1].edge_index is not None
        assert graphs[1].edge_index.shape == (2, 0)
        assert graphs[1].num_edges == 0

    def test_graph_level_y_is_split_per_graph(self) -> None:
        batch = Batch.from_data_list(
            [
                Data(x=jnp.zeros((3, 1)), y=jnp.array(0.0)),
                Data(x=jnp.zeros((2, 1)), y=jnp.array(1.0)),
            ]
        )
        graphs = batch.to_data_list()
        assert graphs[0].y.shape == (1,)
        assert jnp.array_equal(graphs[0].y, jnp.array([0.0]))
        assert jnp.array_equal(graphs[1].y, jnp.array([1.0]))

    def test_graph_level_y_keeps_its_leading_dimension(self) -> None:
        """A per-graph label of shape (1,) survives the round trip unchanged."""
        originals = [
            Data(x=jnp.zeros((3, 1)), y=jnp.array([1.0])),
            Data(x=jnp.zeros((2, 1)), y=jnp.array([2.0])),
        ]
        graphs = Batch.from_data_list(originals).to_data_list()
        assert [graph.y.shape for graph in graphs] == [(1,), (1,)]
        assert graphs == originals

    def test_graph_level_y_with_feature_dimension_keeps_its_rank(self) -> None:
        originals = [
            Data(x=jnp.zeros((3, 1)), y=jnp.zeros((1, 4))),
            Data(x=jnp.zeros((2, 1)), y=jnp.ones((1, 4))),
        ]
        graphs = Batch.from_data_list(originals).to_data_list()
        assert [graph.y.shape for graph in graphs] == [(1, 4), (1, 4)]
        assert graphs == originals

    def test_node_level_y_is_split_by_node_mask(self) -> None:
        batch = Batch.from_data_list(
            [
                Data(x=jnp.zeros((3, 1)), y=jnp.zeros((3,))),
                Data(x=jnp.zeros((2, 1)), y=jnp.ones((2,))),
            ]
        )
        graphs = batch.to_data_list()
        assert jnp.array_equal(graphs[0].y, jnp.zeros((3,)))
        assert jnp.array_equal(graphs[1].y, jnp.ones((2,)))

    def test_unbatched_input_returns_itself(self) -> None:
        batch = Batch(x=jnp.zeros((2, 1)))
        assert batch.to_data_list() == [batch]


class TestPyTree:
    """A Batch must be a pytree whose leaves are all arrays."""

    @staticmethod
    def _batch() -> Batch:
        return Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])])

    def test_leaves_are_arrays(self) -> None:
        leaves = jax.tree_util.tree_leaves(self._batch())
        assert len(leaves) == 4  # x, edge_index, batch, ptr
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_treedef_is_hashable(self) -> None:
        treedef = jax.tree_util.tree_structure(self._batch())
        assert isinstance(hash(treedef), int)

    def test_jit(self) -> None:
        batch = self._batch()
        assert jax.jit(lambda b: b.x.sum())(batch) == batch.x.sum()

    def test_tree_map(self) -> None:
        shapes = jax.tree.map(lambda a: a.shape, self._batch())
        assert isinstance(shapes, Batch)
        assert shapes.x == (5, 1)

    def test_vmap(self) -> None:
        first, second = self._batch(), self._batch()
        stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), first, second)
        sums = jax.vmap(lambda b: b.x.sum())(stacked)
        assert sums.shape == (2,)
        assert jnp.allclose(sums, first.x.sum())

    def test_subclass_leaves_are_arrays(self) -> None:
        batch = FaceBatch.from_data_list(
            [
                FaceData(
                    x=jnp.zeros((3, 2)),
                    face=jnp.array([[0], [1], [2]], dtype=jnp.int32),
                    normal=jnp.ones((1, 3)),
                    label=jnp.array(0.0),
                ),
                FaceData(
                    x=jnp.zeros((2, 2)),
                    face=jnp.array([[0], [1], [0]], dtype=jnp.int32),
                    normal=jnp.ones((1, 3)) * 2,
                    label=jnp.array(1.0),
                ),
            ]
        )
        leaves = jax.tree_util.tree_leaves(batch)
        assert len(leaves) == 6  # x, batch, ptr, face, normal, label
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)
        assert jax.jit(lambda b: b.x.sum())(batch) == 0.0

    def test_config_is_not_a_data_key(self) -> None:
        batch = self._batch()
        assert set(batch.keys()) == {"x", "edge_index", "batch", "ptr"}
        assert "NODE_INDEX_FIELDS" not in batch
        assert FaceBatch.NODE_INDEX_FIELDS == {"face"}
        assert Batch.NODE_INDEX_FIELDS == set()


class TestSubclassBatching:
    """Custom index, element-level and graph-level fields."""

    @staticmethod
    def _meshes() -> list[FaceData]:
        first = FaceData(
            x=jnp.array([[0.0], [1.0], [2.0]]),
            face=jnp.array([[0], [1], [2]], dtype=jnp.int32),
            normal=jnp.array([[1.0, 0.0, 0.0]]),
            label=jnp.array(0.0),
        )
        second = FaceData(
            x=jnp.array([[3.0], [4.0]]),
            face=jnp.array([[0], [1], [0]], dtype=jnp.int32),
            normal=jnp.array([[0.0, 1.0, 0.0]]),
            label=jnp.array(1.0),
        )
        return [first, second]

    def test_index_field_is_offset(self) -> None:
        batch = FaceBatch.from_data_list(self._meshes())
        assert jnp.array_equal(batch.face, jnp.array([[0, 3], [1, 4], [2, 3]]))
        assert batch.normal.shape == (2, 3)
        assert jnp.array_equal(batch.label, jnp.array([0.0, 1.0]))

    def test_round_trip_restores_data_class(self) -> None:
        meshes = self._meshes()
        recovered = FaceBatch.from_data_list(meshes).to_data_list()
        assert len(recovered) == 2
        for original, result in zip(meshes, recovered, strict=True):
            assert isinstance(result, FaceData)
            assert jnp.array_equal(original.x, result.x)
            assert jnp.array_equal(original.face, result.face)
            assert jnp.array_equal(original.normal, result.normal)
            assert original.label == result.label

    def test_mesh_without_faces_batches_next_to_element_attributes(self) -> None:
        """normal aligns with the face axis, so a mesh with no faces may omit it."""
        meshes = self._meshes()
        meshes[1] = meshes[1].replace(face=None, normal=None)
        batch = FaceBatch.from_data_list(meshes)
        assert jnp.array_equal(batch.face, jnp.array([[0], [1], [2]]))
        assert batch.normal.shape == (1, 3)
        assert batch.normal.shape[0] == batch.face.shape[-1]

    def test_element_attribute_missing_on_a_mesh_with_faces_is_rejected(self) -> None:
        meshes = self._meshes()
        meshes[1] = meshes[1].replace(normal=None)
        with pytest.raises(RuntimeError, match="'normal' has 0 rows on graph 1"):
            FaceBatch.from_data_list(meshes)

    def test_element_attribute_row_count_must_match_element_count(self) -> None:
        meshes = self._meshes()
        meshes[0] = meshes[0].replace(normal=jnp.zeros((2, 3)))
        with pytest.raises(RuntimeError, match="'normal' has 2 rows on graph 0"):
            FaceBatch.from_data_list(meshes)

    def test_graph_level_field_must_be_present_everywhere(self) -> None:
        meshes = self._meshes()
        meshes[1] = meshes[1].replace(label=None)
        with pytest.raises(RuntimeError, match="'label'"):
            FaceBatch.from_data_list(meshes)


class TestEquality:
    """Equality compares array fields element-wise instead of truth-testing them."""

    def test_equal_batches_compare_equal(self) -> None:
        graphs = [_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])]
        assert Batch.from_data_list(graphs) == Batch.from_data_list(graphs)

    def test_differing_batches_compare_unequal(self) -> None:
        first = Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])])
        second = Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), _graph(3, [[0], [1]])])
        assert first != second

    def test_batch_differs_from_data(self) -> None:
        assert Batch(x=jnp.zeros((2, 1))) != Data(x=jnp.zeros((2, 1)))

    def test_round_trip_graphs_compare_equal(self) -> None:
        originals = [_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])]
        assert Batch.from_data_list(originals).to_data_list() == originals

    def test_subclass_delegating_eq_compares_element_wise(self) -> None:
        """The documented subclassing recipe keeps == working on array fields."""
        first = FaceBatch(x=jnp.zeros((3, 2)), normal=jnp.ones((1, 3)))
        second = FaceBatch(x=jnp.zeros((3, 2)), normal=jnp.ones((1, 3)))
        assert first == second
        assert first != FaceBatch(x=jnp.zeros((3, 2)), normal=jnp.zeros((1, 3)))
        assert FaceData(x=jnp.zeros((3, 2))) == FaceData(x=jnp.zeros((3, 2)))
        assert FaceData(x=jnp.zeros((3, 2))) != FaceData(x=jnp.ones((3, 2)))
        assert FaceData(x=jnp.zeros((3, 2))) != Data(x=jnp.zeros((3, 2)))


class TestRepr:
    """String representation."""

    def test_repr_reports_batch_size_and_shapes(self) -> None:
        text = repr(Batch.from_data_list([_graph(3, [[0, 1], [1, 2]]), _graph(2, [[0], [1]])]))
        assert text.startswith("Batch(")
        assert "batch_size=2" in text
        assert "x=[5, 1]" in text
        assert "NODE_INDEX_FIELDS" not in text

    def test_repr_counts_trailing_empty_graph(self) -> None:
        batch = Batch.from_data_list(
            [
                _graph(3, [[0, 1], [1, 2]]),
                Data(x=jnp.zeros((0, 1)), edge_index=jnp.zeros((2, 0), dtype=jnp.int32)),
            ]
        )
        assert "batch_size=2" in repr(batch)


def test_primary_index_field_is_deterministic():
    """The primary index field is chosen by name, not by set iteration order.

    ``next(iter(set))`` varies with the per-process hash seed, so the same
    batch either collated or raised depending on the process; alphabetical
    choice is stable.
    """

    @dataclass
    class TwoIndexData(Data):
        face: jnp.ndarray | None = None
        tetra: jnp.ndarray | None = None

    @dataclass
    class TwoIndexBatch(Batch):
        face: jnp.ndarray | None = None
        tetra: jnp.ndarray | None = None

        NODE_INDEX_FIELDS: ClassVar[set[str]] = {"tetra", "face"}
        _DATA_CLASS: ClassVar[type | None] = TwoIndexData

        def __eq__(self, other: object) -> bool:
            return Batch.__eq__(self, other)

    assert TwoIndexBatch._primary_index_field() == "face"
