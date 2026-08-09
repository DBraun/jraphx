"""Tests for edge coalescing converted from PyTorch Geometric to JraphX.

This module tests the coalesce functionality in JraphX for removing duplicate edges.
"""

import jax.numpy as jnp
import pytest

from jraphx.utils.coalesce import coalesce


def test_coalesce_basic():
    """Test basic coalescing without edge attributes."""
    edge_index = jnp.array([[2, 1, 1, 0, 2], [1, 2, 0, 1, 1]])

    out = coalesce(edge_index)
    # Should return sorted unique edges
    expected = jnp.array([[0, 1, 1, 2], [1, 0, 2, 1]])

    # Check that shapes match
    assert out[0].shape == expected.shape
    assert out[1] is None

    # Sort both for comparison since order might differ
    out_sorted = out[0][:, jnp.lexsort([out[0][1], out[0][0]])]
    expected_sorted = expected[:, jnp.lexsort([expected[1], expected[0]])]
    assert jnp.array_equal(out_sorted, expected_sorted)


def test_coalesce_with_edge_attr():
    """Test coalescing with edge attributes."""
    edge_index = jnp.array([[2, 1, 1, 0, 2], [1, 2, 0, 1, 1]])
    edge_attr = jnp.array([[1], [2], [3], [4], [5]])

    out = coalesce(edge_index, edge_attr)

    # Check shapes
    assert out[0].shape[0] == 2  # 2 rows for edge_index
    assert out[0].shape[1] == 4  # 4 unique edges
    assert out[1].shape == (4, 1)  # 4 edges, 1 feature per edge

    # The duplicate edge (2,1) should have summed attributes: 1 + 5 = 6
    # We need to check this more carefully due to sorting
    unique_edges = out[0]
    unique_attrs = out[1]

    # Find the edge (2,1) in the result
    edge_21_mask = (unique_edges[0] == 2) & (unique_edges[1] == 1)
    if jnp.any(edge_21_mask):
        edge_21_attr = unique_attrs[edge_21_mask][0]
        assert jnp.allclose(edge_21_attr, jnp.array([6]))  # 1 + 5


def test_coalesce_with_multiple_edge_attrs():
    """Test coalescing with multiple edge attributes."""
    edge_index = jnp.array([[2, 1, 1, 0, 2], [1, 2, 0, 1, 1]])
    edge_attr = jnp.array([[1], [2], [3], [4], [5]])
    # edge_attr_flat = edge_attr.flatten()  # Not used in this test

    # Test with list-like input (simulated with single attribute)
    out = coalesce(edge_index, edge_attr)
    assert out[0].shape[1] == 4  # 4 unique edges
    assert out[1].shape == (4, 1)


def test_coalesce_without_duplicates():
    """Test coalescing when there are no duplicates."""
    edge_index = jnp.array([[2, 1, 1, 0], [1, 2, 0, 1]])
    edge_attr = jnp.array([[1], [2], [3], [4]])

    out = coalesce(edge_index)
    # Should still return sorted edges
    assert out[0].shape[1] == 4
    assert out[1] is None

    out = coalesce(edge_index, edge_attr)
    assert out[0].shape[1] == 4
    assert out[1].shape == (4, 1)


def test_coalesce_empty():
    """Test coalescing with empty edge_index."""
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)

    out = coalesce(edge_index)
    assert out[0].shape == (2, 0)
    assert out[1] is None

    # With empty edge attributes
    edge_attr = jnp.empty((0, 1))
    out = coalesce(edge_index, edge_attr)
    assert out[0].shape == (2, 0)
    assert out[1].shape == (0, 1)


def test_coalesce_different_reductions():
    """Test coalescing with different reduction operations."""
    # Create edges with duplicates
    edge_index = jnp.array([[0, 0, 1], [1, 1, 2]])  # (0,1) appears twice
    edge_attr = jnp.array([2.0, 3.0, 1.0])

    # Test add reduction (default)
    out_add = coalesce(edge_index, edge_attr, reduce="add")
    # Edge (0,1) should have attribute 2.0 + 3.0 = 5.0

    # Test mean reduction
    out_mean = coalesce(edge_index, edge_attr, reduce="mean")
    # Edge (0,1) should have attribute (2.0 + 3.0) / 2 = 2.5

    # Test max reduction
    out_max = coalesce(edge_index, edge_attr, reduce="max")
    # Edge (0,1) should have attribute max(2.0, 3.0) = 3.0

    # Test min reduction
    out_min = coalesce(edge_index, edge_attr, reduce="min")
    # Edge (0,1) should have attribute min(2.0, 3.0) = 2.0

    # All should have 2 unique edges
    assert out_add[0].shape[1] == 2
    assert out_mean[0].shape[1] == 2
    assert out_max[0].shape[1] == 2
    assert out_min[0].shape[1] == 2


def test_coalesce_large_graph():
    """Test coalescing with a larger graph."""
    # Create a larger edge set with some duplicates
    edge_index = jnp.array(
        [[0, 1, 2, 3, 4, 0, 1, 2], [1, 2, 3, 4, 0, 1, 2, 3]]  # Some duplicates: (0,1), (1,2), (2,3)
    )

    out = coalesce(edge_index)

    # Should have fewer edges than input due to deduplication
    assert out[0].shape[1] <= edge_index.shape[1]

    # All edges should be unique after coalescing
    edge_tuples = [(int(out[0][0, i]), int(out[0][1, i])) for i in range(out[0].shape[1])]
    assert len(edge_tuples) == len(set(edge_tuples))  # All unique


def test_coalesce_high_node_ids():
    """Node ids far beyond the int32 packing threshold must survive intact."""
    edge_index = jnp.array([[0, 99999, 99999], [1, 3, 3]], dtype=jnp.int32)
    edge_attr = jnp.array([1.0, 2.0, 3.0])

    out_index, out_attr = coalesce(edge_index, edge_attr, num_nodes=100000)

    assert jnp.array_equal(out_index, jnp.array([[0, 99999], [1, 3]], dtype=jnp.int32))
    assert jnp.allclose(out_attr, jnp.array([1.0, 5.0]))

    # The same answer without an explicit num_nodes.
    out_index, out_attr = coalesce(edge_index, edge_attr)
    assert jnp.array_equal(out_index, jnp.array([[0, 99999], [1, 3]], dtype=jnp.int32))
    assert jnp.allclose(out_attr, jnp.array([1.0, 5.0]))


def test_coalesce_reduce_sum_alias():
    """``reduce='sum'`` is accepted and behaves like ``'add'``."""
    edge_index = jnp.array([[0, 0, 1], [1, 1, 2]])
    edge_attr = jnp.array([[1.0], [2.0], [3.0]])

    index_sum, attr_sum = coalesce(edge_index, edge_attr, reduce="sum")
    index_add, attr_add = coalesce(edge_index, edge_attr, reduce="add")

    assert jnp.array_equal(index_sum, index_add)
    assert jnp.allclose(attr_sum, attr_add)
    assert jnp.allclose(attr_sum, jnp.array([[3.0], [3.0]]))


def test_coalesce_unknown_reduce():
    """An unsupported reduction fails loudly."""
    edge_index = jnp.array([[0, 0], [1, 1]])
    edge_attr = jnp.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Unknown reduce operation"):
        coalesce(edge_index, edge_attr, reduce="mul")


def test_coalesce_rejects_out_of_range_num_nodes():
    """``num_nodes`` is validated instead of being silently ignored."""
    edge_index = jnp.array([[0, 5], [1, 2]])

    with pytest.raises(ValueError, match="out of range"):
        coalesce(edge_index, num_nodes=3)


def test_coalesce_reduction_values():
    """Each reduction merges duplicate edge attributes correctly."""
    edge_index = jnp.array([[0, 0, 1], [1, 1, 2]])  # (0,1) appears twice
    edge_attr = jnp.array([2.0, 3.0, 1.0])

    expected = {"add": 5.0, "mean": 2.5, "max": 3.0, "min": 2.0}
    for reduce_op, value in expected.items():
        out_index, out_attr = coalesce(edge_index, edge_attr, reduce=reduce_op)
        assert jnp.array_equal(out_index, jnp.array([[0, 1], [1, 2]]))
        assert jnp.allclose(out_attr, jnp.array([value, 1.0]))


def test_coalesce_is_row_sorted():
    """The coalesced edge list is lexicographically sorted, as in PyG."""
    edge_index = jnp.array([[2, 1, 1, 0, 2], [1, 2, 0, 1, 1]])

    out_index, _ = coalesce(edge_index)

    assert jnp.array_equal(out_index, jnp.array([[0, 1, 1, 2], [1, 0, 2, 1]]))


# TODO: The following features from PyG are not supported in JraphX:
# - JIT scripting (torch.jit.script) - JAX has its own JIT compilation
# - Tuple input format for edge_index - JraphX expects arrays
# - Complex edge attribute handling with nested lists - simplified to arrays
#
# The core coalesce functionality for removing duplicate edges and aggregating
# edge attributes is fully supported with multiple reduction operations.


if __name__ == "__main__":
    # Run basic tests
    test_coalesce_basic()
    test_coalesce_with_edge_attr()
    test_coalesce_with_multiple_edge_attrs()
    test_coalesce_without_duplicates()
    test_coalesce_empty()
    test_coalesce_different_reductions()
    test_coalesce_large_graph()
    print("All coalesce tests passed!")
