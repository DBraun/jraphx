"""Tests for self-loop utilities converted from PyTorch Geometric to JraphX.

This module tests the self-loop functionality in JraphX for adding and removing
self-loops from graphs.
"""

import jax.numpy as jnp

from jraphx.utils.loop import add_remaining_self_loops, add_self_loops, remove_self_loops


def test_add_self_loops_basic():
    """Test basic self-loop addition."""
    edge_index = jnp.array([[0, 1, 0], [1, 0, 0]])  # Has one self-loop (0,0)

    out = add_self_loops(edge_index)
    expected = jnp.array([[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]])

    assert out[0].shape == expected.shape
    assert out[1] is None


def test_add_self_loops_with_edge_weight():
    """Test self-loop addition with edge weights."""
    edge_index = jnp.array([[0, 1, 0], [1, 0, 0]])
    edge_weight = jnp.array([0.5, 0.5, 0.5])

    out = add_self_loops(edge_index, edge_weight)
    expected_index = jnp.array([[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]])
    expected_weight = jnp.array([0.5, 0.5, 0.5, 1.0, 1.0])

    assert out[0].shape == expected_index.shape
    assert jnp.allclose(out[1], expected_weight)


def test_add_self_loops_custom_fill():
    """Test self-loop addition with custom fill value."""
    edge_index = jnp.array([[0, 1, 0], [1, 0, 0]])
    edge_weight = jnp.array([0.5, 0.5, 0.5])

    out = add_self_loops(edge_index, edge_weight, fill_value=5.0)
    expected_weight = jnp.array([0.5, 0.5, 0.5, 5.0, 5.0])

    assert jnp.allclose(out[1], expected_weight)


def test_add_self_loops_multidimensional_attr():
    """Test self-loop addition with multi-dimensional edge attributes."""
    edge_index = jnp.array([[0, 1], [1, 0]])
    edge_attr = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 2 edges, 2 features each

    out = add_self_loops(edge_index, edge_attr, fill_value=1.0)
    expected_index = jnp.array([[0, 1, 0, 1], [1, 0, 0, 1]])
    expected_attr = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0, 1.0],  # self-loop for node 0
            [1.0, 1.0],  # self-loop for node 1
        ]
    )

    assert out[0].shape == expected_index.shape
    assert jnp.allclose(out[1], expected_attr)


def test_add_self_loops_empty():
    """Test self-loop addition with empty graph."""
    edge_index = jnp.empty((2, 0), dtype=jnp.int32)
    edge_attr = jnp.empty((0, 2))

    # Should handle empty case gracefully
    out = add_self_loops(edge_index, edge_attr, num_nodes=3)

    # Should add self-loops for all 3 nodes
    expected_index = jnp.array([[0, 1, 2], [0, 1, 2]])
    expected_attr = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    assert jnp.array_equal(out[0], expected_index)
    assert jnp.allclose(out[1], expected_attr)


def test_remove_self_loops():
    """Test self-loop removal."""
    edge_index = jnp.array([[0, 1, 0], [1, 0, 0]])  # (0,0) is self-loop
    edge_attr = jnp.array([[1, 2], [3, 4], [5, 6]])

    expected = jnp.array([[0, 1], [1, 0]])

    out = remove_self_loops(edge_index)
    assert jnp.array_equal(out[0], expected)
    assert out[1] is None

    out = remove_self_loops(edge_index, edge_attr)
    assert jnp.array_equal(out[0], expected)
    assert jnp.array_equal(out[1], jnp.array([[1, 2], [3, 4]]))


def test_remove_self_loops_none():
    """Test self-loop removal when no self-loops exist."""
    edge_index = jnp.array([[0, 1], [1, 2]])
    edge_attr = jnp.array([[1, 2], [3, 4]])

    out = remove_self_loops(edge_index, edge_attr)

    # Should return unchanged since no self-loops
    assert jnp.array_equal(out[0], edge_index)
    assert jnp.array_equal(out[1], edge_attr)


def test_add_remaining_self_loops():
    """Test adding self-loops only for nodes that don't have them."""
    edge_index = jnp.array([[0, 1, 0], [1, 0, 0]])  # (0,0) already exists
    edge_weight = jnp.array([0.5, 0.5, 0.5])

    expected = jnp.array([[0, 1, 0, 1], [1, 0, 0, 1]])  # Only add (1,1)

    out = add_remaining_self_loops(edge_index, edge_weight)
    assert out[0].shape == expected.shape
    # Should have added self-loop for node 1 only
    assert jnp.allclose(out[1], jnp.array([0.5, 0.5, 0.5, 1.0]))


def test_add_remaining_self_loops_none_existing():
    """Test adding remaining self-loops when none exist."""
    edge_index = jnp.array([[0, 1], [1, 0]])
    edge_weight = jnp.array([0.5, 0.5])

    expected = jnp.array([[0, 1, 0, 1], [1, 0, 0, 1]])

    out = add_remaining_self_loops(edge_index, edge_weight)
    assert out[0].shape == expected.shape
    # Should add self-loops for both nodes
    assert jnp.allclose(out[1], jnp.array([0.5, 0.5, 1.0, 1.0]))


def test_add_remaining_self_loops_custom_fill():
    """Test adding remaining self-loops with custom fill value."""
    edge_index = jnp.array([[0, 1, 0], [1, 0, 0]])  # (0,0) already exists
    edge_weight = jnp.array([0.5, 0.5, 0.5])

    out = add_remaining_self_loops(edge_index, edge_weight, fill_value=2.0)
    # Should add (1,1) with weight 2.0
    assert jnp.allclose(out[1], jnp.array([0.5, 0.5, 0.5, 2.0]))


def test_add_remaining_self_loops_multidim():
    """Test adding remaining self-loops with multi-dimensional attributes."""
    edge_index = jnp.array([[0, 1, 0], [1, 0, 0]])  # (0,0) already exists
    edge_attr = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    out = add_remaining_self_loops(edge_index, edge_attr, fill_value=0.5)

    expected_attr = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.5, 0.5]])  # Added for node 1

    assert jnp.allclose(out[1], expected_attr)


def test_loop_functions_consistency():
    """Test consistency between different loop functions."""
    edge_index = jnp.array([[0, 1], [1, 2]])  # No self-loops

    # Adding all self-loops then removing should give original
    with_loops, _ = add_self_loops(edge_index, num_nodes=3)
    without_loops, _ = remove_self_loops(with_loops)

    # Sort both for comparison
    orig_sorted = edge_index[:, jnp.lexsort([edge_index[1], edge_index[0]])]
    result_sorted = without_loops[:, jnp.lexsort([without_loops[1], without_loops[0]])]

    assert jnp.array_equal(orig_sorted, result_sorted)


def test_add_self_loops_string_fill_value():
    """Test add_self_loops with string aggregation fill values."""
    # Create a more complex graph for meaningful aggregation
    edge_index = jnp.array(
        [[0, 1, 2, 0, 1], [1, 2, 0, 0, 1]]
    )  # 0->1, 1->2, 2->0, 0->0 (self), 1->1 (self)
    edge_attr = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Different values per edge

    # Test 'mean' fill value
    edge_index_mean, edge_attr_mean = add_self_loops(
        edge_index, edge_attr, fill_value="mean", num_nodes=3
    )

    # Expected:
    # Node 0 receives edges with attrs [4.0, 3.0] -> mean = 3.5
    # Node 1 receives edges with attrs [1.0, 5.0] -> mean = 3.0
    # Node 2 receives edges with attrs [2.0] -> mean = 2.0
    expected_self_loop_attrs = jnp.array([[3.5], [3.0], [2.0]])

    # The last 3 entries should be the self-loop attributes
    assert jnp.allclose(edge_attr_mean[-3:], expected_self_loop_attrs, atol=1e-6)

    # Test 'add' fill value
    edge_index_add, edge_attr_add = add_self_loops(
        edge_index, edge_attr, fill_value="add", num_nodes=3
    )
    expected_add_attrs = jnp.array([[7.0], [6.0], [2.0]])  # sum of incoming edges
    assert jnp.allclose(edge_attr_add[-3:], expected_add_attrs)

    # Test 'max' fill value
    edge_index_max, edge_attr_max = add_self_loops(
        edge_index, edge_attr, fill_value="max", num_nodes=3
    )
    expected_max_attrs = jnp.array([[4.0], [5.0], [2.0]])  # max of incoming edges
    assert jnp.allclose(edge_attr_max[-3:], expected_max_attrs)

    # Test 'min' fill value
    edge_index_min, edge_attr_min = add_self_loops(
        edge_index, edge_attr, fill_value="min", num_nodes=3
    )
    expected_min_attrs = jnp.array([[3.0], [1.0], [2.0]])  # min of incoming edges
    assert jnp.allclose(edge_attr_min[-3:], expected_min_attrs)


def test_add_self_loops_string_empty_graph():
    """Test string fill_value with graphs that have no incoming edges for some nodes."""
    edge_index = jnp.array([[0, 1], [1, 2]])  # Node 0 has no incoming edges
    edge_attr = jnp.array([[1.0], [2.0]])

    # Test with 'mean' - nodes with no incoming edges should get 0
    edge_index_new, edge_attr_new = add_self_loops(
        edge_index, edge_attr, fill_value="mean", num_nodes=3
    )

    # Node 0: no incoming -> 0, Node 1: [1.0] -> 1.0, Node 2: [2.0] -> 2.0
    expected_attrs = jnp.array([[0.0], [1.0], [2.0]])
    assert jnp.allclose(edge_attr_new[-3:], expected_attrs)


# TODO: The following features from PyG are not yet implemented in JraphX:
# - contains_self_loops() function - can be implemented using simple comparison
# - segregate_self_loops() function - would separate self-loops from other edges
# - get_self_loop_attr() function - would extract attributes of self-loops only
# - EdgeIndex class support - JraphX uses plain arrays
# - Sparse tensor (torch.sparse.FloatTensor) support - JAX uses dense arrays
# - "add" fill_value mode - would add to existing self-loop weights
# - Bipartite graph support with num_nodes tuple - would need separate implementation
#
# The core functionality for adding and removing self-loops is implemented
# and working correctly with JAX arrays and attributes.


if __name__ == "__main__":
    # Run basic tests
    test_add_self_loops_basic()
    test_add_self_loops_with_edge_weight()
    test_add_self_loops_custom_fill()
    test_add_self_loops_multidimensional_attr()
    test_add_self_loops_empty()
    test_remove_self_loops()
    test_remove_self_loops_none()
    test_add_remaining_self_loops()
    test_add_remaining_self_loops_none_existing()
    test_add_remaining_self_loops_custom_fill()
    test_add_remaining_self_loops_multidim()
    test_loop_functions_consistency()
    print("All loop tests passed!")
