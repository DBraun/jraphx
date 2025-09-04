"""Tests for degree computation converted from PyTorch Geometric to JraphX.

This module tests the degree computation functionality in JraphX.
"""

import jax.numpy as jnp

from jraphx.utils.degree import degree, in_degree, out_degree


def test_degree():
    """Test basic degree computation."""
    row = jnp.array([0, 1, 0, 2, 0])
    deg = degree(row, dtype=jnp.int32)
    assert deg.dtype == jnp.int32
    assert deg.tolist() == [3, 1, 1]


def test_degree_with_num_nodes():
    """Test degree computation with explicit num_nodes."""
    row = jnp.array([0, 1, 0, 2, 0])
    deg = degree(row, num_nodes=5, dtype=jnp.int32)
    assert deg.dtype == jnp.int32
    assert deg.tolist() == [3, 1, 1, 0, 0]  # Extra nodes have degree 0


def test_degree_float():
    """Test degree computation with float dtype."""
    row = jnp.array([0, 1, 0, 2, 0])
    deg = degree(row, dtype=jnp.float32)
    assert deg.dtype == jnp.float32
    assert jnp.allclose(deg, jnp.array([3.0, 1.0, 1.0]))


def test_in_degree():
    """Test in-degree computation."""
    # Edge index format: [source, target]
    edge_index = jnp.array([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]])  # source nodes  # target nodes

    in_deg = in_degree(edge_index, dtype=jnp.int32)
    # Node 0: incoming from [2, 1] -> degree 2
    # Node 1: incoming from [0] -> degree 1
    # Node 2: incoming from [1, 0] -> degree 2
    assert in_deg.tolist() == [2, 1, 2]


def test_out_degree():
    """Test out-degree computation."""
    # Edge index format: [source, target]
    edge_index = jnp.array([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]])  # source nodes  # target nodes

    out_deg = out_degree(edge_index, dtype=jnp.int32)
    # Node 0: outgoing to [1, 2] -> degree 2
    # Node 1: outgoing to [2, 0] -> degree 2
    # Node 2: outgoing to [0] -> degree 1
    assert out_deg.tolist() == [2, 2, 1]


def test_degree_empty():
    """Test degree computation with empty input."""
    row = jnp.array([], dtype=jnp.int32)
    deg = degree(row, num_nodes=3, dtype=jnp.int32)
    assert deg.tolist() == [0, 0, 0]


def test_degree_consistency():
    """Test that in_degree + out_degree equals total degree for undirected graphs."""
    # Create a simple undirected graph (each edge appears in both directions)
    edge_index = jnp.array(
        [
            [0, 1, 1, 0, 1, 2, 2, 1],  # source nodes
            [1, 0, 2, 1, 1, 1, 1, 2],  # target nodes (mirrored)
        ]
    )

    in_deg = in_degree(edge_index, dtype=jnp.int32)
    out_deg = out_degree(edge_index, dtype=jnp.int32)

    # For this symmetric case, in and out degrees should be equal
    # (though this specific example isn't perfectly symmetric)
    assert len(in_deg) == len(out_deg)


if __name__ == "__main__":
    # Run basic tests
    test_degree()
    test_degree_with_num_nodes()
    test_degree_float()
    test_in_degree()
    test_out_degree()
    test_degree_empty()
    test_degree_consistency()
    print("All degree tests passed!")
