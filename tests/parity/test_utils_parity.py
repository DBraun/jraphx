"""Numerical parity of jraphx utilities against an installed torch_geometric.

Every test in this package requires ``torch`` and ``torch_geometric`` and is
skipped when they are absent, so the regular suite is unaffected. CI runs them
in a dedicated job with CPU torch installed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyg_utils = pytest.importorskip("torch_geometric.utils")

from torch_geometric.nn import (  # noqa: E402
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import SortAggregation  # noqa: E402

import jraphx.nn.pool as jpool  # noqa: E402
from jraphx.utils import (  # noqa: E402
    coalesce,
    degree,
    scatter,
    scatter_softmax,
    to_undirected,
)
from jraphx.utils.loop import (  # noqa: E402
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
)


def _np(x) -> np.ndarray:
    """Convert a torch tensor or jax array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _assert_close(jax_result, torch_result, atol=1e-6):
    np.testing.assert_allclose(_np(jax_result), _np(torch_result), atol=atol, rtol=1e-5)


@pytest.fixture
def edge_case_values():
    """Values with duplicates, negatives and an empty segment."""
    rng = np.random.default_rng(0)
    src = rng.normal(size=(11, 3)).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3])  # segment 2 is empty
    return src, index


@pytest.mark.parametrize("reduce", ["add", "mean", "max", "min"])
def test_scatter_matches_pyg(edge_case_values, reduce):
    src, index = edge_case_values
    ours = scatter(jnp.asarray(src), jnp.asarray(index), dim_size=4, reduce=reduce)
    theirs = pyg_utils.scatter(
        torch.tensor(src), torch.tensor(index), dim=0, dim_size=4, reduce=reduce
    )
    _assert_close(ours, theirs)


def test_scatter_softmax_matches_pyg(edge_case_values):
    src, index = edge_case_values
    ours = scatter_softmax(jnp.asarray(src[:, 0]), jnp.asarray(index), dim_size=4)
    theirs = pyg_utils.softmax(torch.tensor(src[:, 0]), torch.tensor(index), num_nodes=4)
    _assert_close(ours, theirs)


@pytest.mark.parametrize("reduce", ["add", "mean", "max"])
def test_coalesce_matches_pyg(reduce):
    edge_index = np.array([[1, 0, 1, 2, 1], [0, 1, 0, 1, 0]])
    edge_attr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    our_ei, our_ea = coalesce(
        jnp.asarray(edge_index), jnp.asarray(edge_attr), num_nodes=3, reduce=reduce
    )
    their_ei, their_ea = pyg_utils.coalesce(
        torch.tensor(edge_index), torch.tensor(edge_attr), num_nodes=3, reduce=reduce
    )
    _assert_close(our_ei, their_ei)
    _assert_close(our_ea, their_ea)


@pytest.mark.parametrize("reduce", ["add", "mean", "max"])
def test_to_undirected_matches_pyg(reduce):
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 0]])
    edge_attr = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    our_ei, our_ea = to_undirected(
        jnp.asarray(edge_index), jnp.asarray(edge_attr), num_nodes=3, reduce=reduce
    )
    their_ei, their_ea = pyg_utils.to_undirected(
        torch.tensor(edge_index), torch.tensor(edge_attr), num_nodes=3, reduce=reduce
    )
    _assert_close(our_ei, their_ei)
    _assert_close(our_ea, their_ea)


def test_degree_matches_pyg():
    row = np.array([0, 1, 0, 2, 0])
    ours = degree(jnp.asarray(row), num_nodes=4)
    theirs = pyg_utils.degree(torch.tensor(row), num_nodes=4)
    _assert_close(ours, theirs)


def test_add_self_loops_matches_pyg_float_fill():
    edge_index = np.array([[0, 1], [1, 2]])
    edge_attr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    our_ei, our_ea = add_self_loops(
        jnp.asarray(edge_index), jnp.asarray(edge_attr), fill_value=0.5, num_nodes=3
    )
    their_ei, their_ea = pyg_utils.add_self_loops(
        torch.tensor(edge_index), torch.tensor(edge_attr), fill_value=0.5, num_nodes=3
    )
    _assert_close(our_ei, their_ei)
    _assert_close(our_ea, their_ea)


def test_add_self_loops_matches_pyg_mean_fill():
    edge_index = np.array([[0, 1, 2], [1, 1, 2]])
    edge_attr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    our_ei, our_ea = add_self_loops(
        jnp.asarray(edge_index), jnp.asarray(edge_attr), fill_value="mean", num_nodes=3
    )
    their_ei, their_ea = pyg_utils.add_self_loops(
        torch.tensor(edge_index), torch.tensor(edge_attr), fill_value="mean", num_nodes=3
    )
    _assert_close(our_ei, their_ei)
    _assert_close(our_ea, their_ea)


def test_remove_self_loops_matches_pyg():
    edge_index = np.array([[0, 0, 1, 2], [0, 1, 1, 0]])
    edge_attr = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    our_ei, our_ea = remove_self_loops(jnp.asarray(edge_index), jnp.asarray(edge_attr))
    their_ei, their_ea = pyg_utils.remove_self_loops(
        torch.tensor(edge_index), torch.tensor(edge_attr)
    )
    _assert_close(our_ei, their_ei)
    _assert_close(our_ea, their_ea)


def test_add_remaining_self_loops_matches_pyg():
    # Node 0 carries a duplicated loop; node 1 has none.
    edge_index = np.array([[0, 0, 1], [0, 0, 2]])
    edge_attr = np.array([10.0, 20.0, 3.0], dtype=np.float32)

    our_ei, our_ea = add_remaining_self_loops(
        jnp.asarray(edge_index), jnp.asarray(edge_attr), fill_value=1.0, num_nodes=3
    )
    their_ei, their_ea = pyg_utils.add_remaining_self_loops(
        torch.tensor(edge_index), torch.tensor(edge_attr), fill_value=1.0, num_nodes=3
    )
    _assert_close(our_ei, their_ei)
    _assert_close(our_ea, their_ea)


def test_global_pools_match_pyg():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 4)).astype(np.float32)
    batch = np.array([0, 0, 0, 1, 1, 2, 2])

    for ours_fn, theirs_fn in [
        (jpool.global_add_pool, global_add_pool),
        (jpool.global_mean_pool, global_mean_pool),
        (jpool.global_max_pool, global_max_pool),
    ]:
        ours = ours_fn(jnp.asarray(x), jnp.asarray(batch), size=3)
        theirs = theirs_fn(torch.tensor(x), torch.tensor(batch), size=3)
        _assert_close(ours, theirs)


def test_global_sort_pool_matches_pyg():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(9, 3)).astype(np.float32)
    batch = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

    ours = jpool.global_sort_pool(jnp.asarray(x), jnp.asarray(batch), k=4, size=2)
    theirs = SortAggregation(k=4)(torch.tensor(x), torch.tensor(batch))
    _assert_close(ours, theirs)


def test_global_sort_pool_matches_pyg_with_padding():
    # Fewer nodes than k, all sort scores negative: padding must sort last.
    x = np.array([[-1.0, -5.0], [-2.0, -3.0]], dtype=np.float32)
    batch = np.array([0, 0])

    ours = jpool.global_sort_pool(jnp.asarray(x), jnp.asarray(batch), k=3, size=1)
    theirs = SortAggregation(k=3)(torch.tensor(x), torch.tensor(batch))
    _assert_close(ours, theirs)
