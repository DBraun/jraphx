"""Tests for hierarchical pooling layers (TopKPooling, SAGPooling)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jraphx.nn.pool import SAGPooling, TopKPooling

# A 4 node path graph 0 -> 1 -> 2 -> 3 (plus reverse edges).
EDGE_INDEX = jnp.array([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
X = jnp.array([[4.0, 1.0], [3.0, 1.0], [2.0, 1.0], [1.0, 1.0]])


def _deterministic_pool(**kwargs) -> TopKPooling:
    """Build a TopKPooling layer whose projection is the first feature axis.

    Args:
        **kwargs: Forwarded to :class:`TopKPooling`.

    Returns:
        A pooling layer with ``weight = [[1, 0]]``, so the raw score of a node is its
        first feature.
    """
    pool = TopKPooling(2, rngs=nnx.Rngs(0), **kwargs)
    pool.weight[...] = jnp.array([[1.0, 0.0]])
    return pool


def test_topk_pool_gates_features_by_score() -> None:
    """Selected features are multiplied by tanh of their score at multiplier 1.0."""
    pool = _deterministic_pool(ratio=0.5)
    out, _, _, _, perm = pool(X, EDGE_INDEX)

    assert perm.tolist() == [0, 1]
    expected = X[perm] * jnp.tanh(jnp.array([4.0, 3.0])).reshape(-1, 1)
    assert jnp.allclose(out, expected)


def test_topk_pool_multiplier_scales_gated_features() -> None:
    """The multiplier scales the gated features rather than replacing the gate."""
    gated = _deterministic_pool(ratio=0.5)(X, EDGE_INDEX)[0]
    scaled = _deterministic_pool(ratio=0.5, multiplier=3.0)(X, EDGE_INDEX)[0]
    assert jnp.allclose(scaled, 3.0 * gated)


def test_topk_pool_gradient_flows_to_projection() -> None:
    """The projection vector receives a nonzero gradient at the default multiplier."""
    pool = _deterministic_pool(ratio=0.5)

    def loss_fn(module: TopKPooling) -> jnp.ndarray:
        return module(X, EDGE_INDEX)[0].sum()

    grads = jax.tree.leaves(nnx.grad(loss_fn)(pool))
    assert len(grads) == 1
    assert jnp.any(grads[0] != 0.0)


def test_topk_pool_score_is_normalized_by_projection_norm() -> None:
    """Scaling the projection vector leaves the scores (and hence the output) unchanged."""
    pool = _deterministic_pool(ratio=0.5)
    out = pool(X, EDGE_INDEX)[0]

    pool.weight[...] = pool.weight[...] * 7.5
    scaled_out = pool(X, EDGE_INDEX)[0]

    assert jnp.allclose(out, scaled_out, atol=1e-5)


def test_topk_pool_ratio_rounds_up() -> None:
    """A float ratio keeps ceil(ratio * num_nodes) nodes."""
    x = jnp.array([[3.0, 1.0], [2.0, 1.0], [1.0, 1.0]])
    edge_index = jnp.array([[0, 1], [1, 2]])

    out, _, _, _, perm = _deterministic_pool(ratio=0.5)(x, edge_index)
    assert out.shape == (2, 2)
    assert perm.tolist() == [0, 1]

    out_all, _, _, _, _ = _deterministic_pool(ratio=1.0)(x, edge_index)
    assert out_all.shape == (3, 2)


def test_topk_pool_integer_ratio_is_per_graph() -> None:
    """An integer ratio keeps that many nodes in every graph, not globally."""
    x = jnp.array([[4.0, 1.0], [3.0, 1.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    edge_index = jnp.array([[0, 3], [1, 4]])
    batch = jnp.array([0, 0, 0, 1, 1])

    out, _, _, pooled_batch, perm = _deterministic_pool(ratio=1)(x, edge_index, batch=batch)

    assert out.shape == (2, 2)
    assert perm.tolist() == [0, 3]
    assert pooled_batch.tolist() == [0, 1]


def test_topk_pool_float_ratio_is_per_graph() -> None:
    """A float ratio keeps ceil(ratio * N_i) nodes in each graph."""
    x = jnp.array([[4.0, 1.0], [3.0, 1.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    edge_index = jnp.array([[0, 3], [1, 4]])
    batch = jnp.array([0, 0, 0, 1, 1])

    _, _, _, pooled_batch, perm = _deterministic_pool(ratio=0.5)(x, edge_index, batch=batch)

    # ceil(0.5 * 3) = 2 nodes for graph 0, ceil(0.5 * 2) = 1 node for graph 1.
    assert perm.tolist() == [0, 1, 3]
    assert pooled_batch.tolist() == [0, 0, 1]


def test_topk_pool_edges_are_filtered_and_relabeled() -> None:
    """Only edges between selected nodes survive, with indices remapped to [0, k)."""
    edge_attr = jnp.arange(EDGE_INDEX.shape[1], dtype=jnp.float32).reshape(-1, 1)
    pool = _deterministic_pool(ratio=0.5)

    _, pooled_edge_index, pooled_edge_attr, _, _ = pool(X, EDGE_INDEX, edge_attr=edge_attr)

    # Nodes 0 and 1 survive; edges (0, 1) and (1, 0) connect them.
    assert pooled_edge_index.tolist() == [[0, 1], [1, 0]]
    assert pooled_edge_attr.tolist() == [[0.0], [3.0]]


def test_topk_pool_min_score_uses_softmax_scores() -> None:
    """min_score thresholds softmax-normalized scores and gates with them."""
    x = jnp.array([[3.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    edge_index = jnp.array([[0, 1], [1, 2]])
    pool = _deterministic_pool(ratio=0.5, min_score=0.5)

    out, _, _, _, perm = pool(x, edge_index)

    softmax_scores = jax.nn.softmax(jnp.array([3.0, 1.0, 0.0]))
    assert perm.tolist() == [0]
    assert jnp.allclose(out, x[:1] * softmax_scores[0])


def test_topk_pool_min_score_keeps_best_node() -> None:
    """A min_score above every score still keeps the best node of each graph."""
    x = jnp.array([[3.0, 1.0], [1.0, 1.0], [0.0, 1.0], [2.0, 1.0]])
    edge_index = jnp.array([[0, 2], [1, 3]])
    batch = jnp.array([0, 0, 1, 1])
    pool = _deterministic_pool(min_score=0.99)

    _, _, _, pooled_batch, perm = pool(x, edge_index, batch=batch)

    assert perm.tolist() == [0, 3]
    assert pooled_batch.tolist() == [0, 1]


def test_topk_pool_min_score_ignores_ratio() -> None:
    """With min_score set, more nodes than the ratio allows can be kept."""
    x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    edge_index = jnp.array([[0, 1], [1, 2]])
    pool = _deterministic_pool(ratio=0.25, min_score=0.1)

    _, _, _, _, perm = pool(x, edge_index)

    # All four scores equal 0.25 > 0.1, while ratio=0.25 would keep a single node.
    assert perm.tolist() == [0, 1, 2, 3]


def test_topk_pool_min_score_gradient_flows_to_projection() -> None:
    """The min_score branch also gates features, so the projection is trainable."""
    pool = _deterministic_pool(min_score=0.1)

    def loss_fn(module: TopKPooling) -> jnp.ndarray:
        return module(X, EDGE_INDEX)[0].sum()

    grads = jax.tree.leaves(nnx.grad(loss_fn)(pool))
    assert jnp.any(grads[0] != 0.0)


def test_topk_pool_requires_rngs() -> None:
    """Omitting rngs raises a clear error instead of an AttributeError."""
    with pytest.raises(ValueError, match="requires `rngs`"):
        TopKPooling(4)


def test_topk_pool_rejects_unknown_nonlinearity() -> None:
    """An unsupported nonlinearity is rejected at construction time."""
    with pytest.raises(ValueError, match="Unknown nonlinearity"):
        TopKPooling(4, nonlinearity="relu", rngs=nnx.Rngs(0))


def test_topk_pool_sigmoid_nonlinearity() -> None:
    """The sigmoid nonlinearity gates with sigmoid scores."""
    pool = _deterministic_pool(ratio=0.5, nonlinearity="sigmoid")
    out, _, _, _, perm = pool(X, EDGE_INDEX)

    expected = X[perm] * jax.nn.sigmoid(jnp.array([4.0, 3.0])).reshape(-1, 1)
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize("gnn", ["gcn", "gat", "sage"])
def test_sag_pool_shapes(gnn: str) -> None:
    """SAGPooling coarsens the graph for every supported scoring GNN."""
    pool = SAGPooling(2, ratio=0.5, gnn=gnn, rngs=nnx.Rngs(0))
    out, pooled_edge_index, _, _, perm = pool(X, EDGE_INDEX)

    assert out.shape == (2, 2)
    assert perm.shape == (2,)
    assert pooled_edge_index.shape[0] == 2
    assert jnp.all(pooled_edge_index < 2)


@pytest.mark.parametrize("gnn", ["gcn", "gat", "sage"])
def test_sag_pool_gradient_flows_to_scorer(gnn: str) -> None:
    """The scoring GNN receives a nonzero gradient at the default multiplier."""
    pool = SAGPooling(2, ratio=0.5, gnn=gnn, rngs=nnx.Rngs(0))

    def loss_fn(module: SAGPooling) -> jnp.ndarray:
        return module(X, EDGE_INDEX)[0].sum()

    grads = jax.tree.leaves(nnx.grad(loss_fn)(pool))
    assert len(grads) > 0
    assert any(jnp.any(grad != 0.0) for grad in grads)


def test_sag_pool_has_no_unused_projection() -> None:
    """SAGPooling scores with its GNN and does not carry a dead projection vector."""
    pool = SAGPooling(2, ratio=0.5, rngs=nnx.Rngs(0))
    assert not hasattr(pool, "weight")


def test_sag_pool_requires_rngs() -> None:
    """Omitting rngs raises a clear error instead of an AttributeError."""
    with pytest.raises(ValueError, match="requires `rngs`"):
        SAGPooling(4)


def test_sag_pool_rejects_unknown_gnn() -> None:
    """An unsupported scoring GNN is rejected at construction time."""
    with pytest.raises(ValueError, match="Unknown GNN type"):
        SAGPooling(4, gnn="mlp", rngs=nnx.Rngs(0))


def test_sag_pool_min_score_is_per_graph() -> None:
    """SAGPooling shares the min_score selection rule with TopKPooling."""
    batch = jnp.array([0, 0, 1, 1])
    pool = SAGPooling(2, min_score=0.99, rngs=nnx.Rngs(0))

    _, _, _, pooled_batch, perm = pool(X, EDGE_INDEX, batch=batch)

    assert perm.shape == (2,)
    assert pooled_batch.tolist() == [0, 1]
