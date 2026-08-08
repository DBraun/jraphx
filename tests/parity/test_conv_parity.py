"""Numerical parity of jraphx convolution layers against torch_geometric.

Each test builds the PyG layer, transplants its parameters into the jraphx
layer, runs both on the same graph and compares outputs elementwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

torch = pytest.importorskip("torch")
pyg_nn = pytest.importorskip("torch_geometric.nn")

from jraphx.nn.conv import (  # noqa: E402
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    SAGEConv,
    TransformerConv,
)


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _assert_close(jax_result, torch_result, atol=1e-5):
    np.testing.assert_allclose(_np(jax_result), _np(torch_result), atol=atol, rtol=1e-4)


def _copy_linear(dst, src):
    """Copy a torch/PyG Linear into a flax nnx.Linear (weight is transposed)."""
    dst.kernel[...] = jnp.asarray(_np(src.weight).T)
    if src.bias is not None:
        assert dst.use_bias
        dst.bias[...] = jnp.asarray(_np(src.bias))


def _graph(
    num_src=6, num_dst=None, num_edges=12, in_features=5, edge_dim=None, seed=0, self_loops=True
):
    """A random graph, returned in both frameworks' representations."""
    rng = np.random.default_rng(seed)
    num_dst = num_src if num_dst is None else num_dst
    x = rng.normal(size=(num_src, in_features)).astype(np.float32)
    edge_index = np.stack(
        [rng.integers(0, num_src, size=num_edges), rng.integers(0, num_dst, size=num_edges)]
    )
    if not self_loops:
        # Shift loop targets by one; only meaningful for square graphs
        loops = edge_index[0] == edge_index[1]
        edge_index[1, loops] = (edge_index[1, loops] + 1) % num_dst
    edge_attr = None
    if edge_dim is not None:
        edge_attr = rng.normal(size=(num_edges, edge_dim)).astype(np.float32)
    return x, edge_index, edge_attr


@pytest.mark.parametrize("improved", [False, True])
def test_gcn_conv_matches_pyg(improved):
    x, edge_index, _ = _graph(in_features=5)
    edge_weight = np.abs(np.random.default_rng(1).normal(size=edge_index.shape[1])).astype(
        np.float32
    )

    theirs = pyg_nn.GCNConv(5, 7, improved=improved)
    ours = GCNConv(5, 7, improved=improved, rngs=nnx.Rngs(0))
    _copy_linear(ours.linear, theirs.lin)
    ours.bias[...] = jnp.asarray(_np(theirs.bias))

    out_theirs = theirs(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_weight))
    out_ours = ours(jnp.asarray(x), jnp.asarray(edge_index), jnp.asarray(edge_weight))
    _assert_close(out_ours, out_theirs)


def test_gcn_conv_matches_pyg_with_existing_self_loops():
    x, _, _ = _graph(num_src=4, in_features=5)
    edge_index = np.array([[0, 0, 1, 2, 3], [0, 1, 2, 3, 0]])  # node 0 already has a loop

    theirs = pyg_nn.GCNConv(5, 7)
    ours = GCNConv(5, 7, rngs=nnx.Rngs(0))
    _copy_linear(ours.linear, theirs.lin)
    ours.bias[...] = jnp.asarray(_np(theirs.bias))

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


@pytest.mark.parametrize("concat", [True, False])
def test_gat_conv_matches_pyg(concat):
    x, edge_index, _ = _graph(in_features=5)

    theirs = pyg_nn.GATConv(5, 4, heads=3, concat=concat)
    ours = GATConv(5, 4, heads=3, concat=concat, rngs=nnx.Rngs(0))
    _copy_linear(ours.lin, theirs.lin)
    ours.att_src[...] = jnp.asarray(_np(theirs.att_src)[0])
    ours.att_dst[...] = jnp.asarray(_np(theirs.att_dst)[0])
    ours.bias[...] = jnp.asarray(_np(theirs.bias))

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


def test_gat_conv_matches_pyg_with_edge_features():
    x, edge_index, edge_attr = _graph(in_features=5, edge_dim=3)

    theirs = pyg_nn.GATConv(5, 4, heads=2, edge_dim=3, fill_value=0.5)
    ours = GATConv(5, 4, heads=2, edge_dim=3, fill_value=0.5, rngs=nnx.Rngs(0))
    _copy_linear(ours.lin, theirs.lin)
    _copy_linear(ours.lin_edge, theirs.lin_edge)
    ours.att_src[...] = jnp.asarray(_np(theirs.att_src)[0])
    ours.att_dst[...] = jnp.asarray(_np(theirs.att_dst)[0])
    ours.att_edge[...] = jnp.asarray(_np(theirs.att_edge)[0])
    ours.bias[...] = jnp.asarray(_np(theirs.bias))

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index), jnp.asarray(edge_attr)),
        theirs(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attr)),
    )


def test_gat_conv_matches_pyg_with_existing_self_loop():
    x, _, _ = _graph(num_src=4, in_features=5)
    edge_index = np.array([[0, 0, 1, 2, 3], [0, 1, 2, 3, 0]])

    theirs = pyg_nn.GATConv(5, 4, heads=2)
    ours = GATConv(5, 4, heads=2, rngs=nnx.Rngs(0))
    _copy_linear(ours.lin, theirs.lin)
    ours.att_src[...] = jnp.asarray(_np(theirs.att_src)[0])
    ours.att_dst[...] = jnp.asarray(_np(theirs.att_dst)[0])
    ours.bias[...] = jnp.asarray(_np(theirs.bias))

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


@pytest.mark.parametrize("share_weights", [False, True])
def test_gatv2_conv_matches_pyg(share_weights):
    x, edge_index, _ = _graph(in_features=5)

    theirs = pyg_nn.GATv2Conv(5, 4, heads=3, share_weights=share_weights)
    ours = GATv2Conv(5, 4, heads=3, share_weights=share_weights, rngs=nnx.Rngs(0))
    _copy_linear(ours.lin_l, theirs.lin_l)
    if not share_weights:
        _copy_linear(ours.lin_r, theirs.lin_r)
    ours.att[...] = jnp.asarray(_np(theirs.att)[0])
    ours.bias[...] = jnp.asarray(_np(theirs.bias))

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


def _make_gatv2_pair(edge_dim=3):
    theirs = pyg_nn.GATv2Conv(5, 4, heads=2, edge_dim=edge_dim)  # fill_value="mean" default
    ours = GATv2Conv(5, 4, heads=2, edge_dim=edge_dim, rngs=nnx.Rngs(0))
    _copy_linear(ours.lin_l, theirs.lin_l)
    _copy_linear(ours.lin_r, theirs.lin_r)
    _copy_linear(ours.lin_edge, theirs.lin_edge)
    ours.att[...] = jnp.asarray(_np(theirs.att)[0])
    ours.bias[...] = jnp.asarray(_np(theirs.bias))
    return ours, theirs


def test_gatv2_conv_matches_pyg_with_edge_features_mean_fill():
    # No pre-existing self-loops: the "mean" fill of an injected loop then
    # reduces over the same edge set in both frameworks.
    x, edge_index, edge_attr = _graph(in_features=5, edge_dim=3, self_loops=False)

    ours, theirs = _make_gatv2_pair()
    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index), jnp.asarray(edge_attr)),
        theirs(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attr)),
    )


@pytest.mark.xfail(
    reason="Documented residual: with a string fill_value and a pre-existing "
    "self-loop, jraphx reduces the loop's generated features over a set that "
    "includes the original loop, which PyG excludes (see the changelog).",
    strict=True,
)
def test_gatv2_conv_mean_fill_with_existing_loop_known_divergence():
    x, edge_index, edge_attr = _graph(in_features=5, edge_dim=3, self_loops=True)
    assert (edge_index[0] == edge_index[1]).any()

    ours, theirs = _make_gatv2_pair()
    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index), jnp.asarray(edge_attr)),
        theirs(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attr)),
    )


@pytest.mark.parametrize("aggr", ["mean", "max"])
def test_sage_conv_matches_pyg(aggr):
    x, edge_index, _ = _graph(in_features=5)

    theirs = pyg_nn.SAGEConv(5, 7, aggr=aggr)
    ours = SAGEConv(5, 7, aggr=aggr, rngs=nnx.Rngs(0))
    # PyG applies lin_l (with bias) to the aggregation and lin_r (no bias) to
    # the root; jraphx holds the single bias on lin_r instead.
    ours.lin.kernel[...] = jnp.asarray(_np(theirs.lin_l.weight).T)
    ours.lin_r.kernel[...] = jnp.asarray(_np(theirs.lin_r.weight).T)
    ours.lin_r.bias[...] = jnp.asarray(_np(theirs.lin_l.bias))

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


def _torch_mlp(sizes, seed=0):
    """A torch Sequential MLP: Linear (ReLU Linear)*."""
    torch.manual_seed(seed)
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=False)):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def _nnx_copy_of_torch_mlp(torch_seq):
    """An nnx.Sequential mirroring a torch Sequential's current weights.

    Must be called *after* the torch module is handed to a PyG layer:
    GINConv/GINEConv/EdgeConv re-initialize their wrapped network in their
    constructors, so weights copied earlier would be stale.
    """
    layers = []
    for i, module in enumerate(torch_seq):
        if isinstance(module, torch.nn.Linear):
            lin = nnx.Linear(module.in_features, module.out_features, rngs=nnx.Rngs(i))
            _copy_linear(lin, module)
            layers.append(lin)
        elif isinstance(module, torch.nn.ReLU):
            layers.append(nnx.relu)
        else:
            raise RuntimeError(f"Unhandled torch module {type(module).__name__}")
    return nnx.Sequential(*layers)


@pytest.mark.parametrize("eps", [0.0, 0.4])
def test_gin_conv_matches_pyg(eps):
    x, edge_index, _ = _graph(in_features=5)

    theirs = pyg_nn.GINConv(_torch_mlp([5, 8, 7]), eps=eps)
    ours = GINConv(_nnx_copy_of_torch_mlp(theirs.nn), eps=eps)

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


def test_gine_conv_matches_pyg():
    x, edge_index, edge_attr = _graph(in_features=5, edge_dim=3)

    theirs = pyg_nn.GINEConv(_torch_mlp([5, 8, 7]), eps=0.2, edge_dim=3)
    ours = GINEConv(_nnx_copy_of_torch_mlp(theirs.nn), eps=0.2, edge_dim=3, rngs=nnx.Rngs(0))
    _copy_linear(ours.lin, theirs.lin)

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index), jnp.asarray(edge_attr)),
        theirs(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attr)),
    )


def test_edge_conv_matches_pyg():
    x, edge_index, _ = _graph(in_features=5)

    theirs = pyg_nn.EdgeConv(_torch_mlp([10, 8, 7]), aggr="max")
    ours = EdgeConv(_nnx_copy_of_torch_mlp(theirs.nn), aggr="max")

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


def _copy_transformer(ours, theirs, heads, out_features):
    """Fuse PyG's separate q/k/v projections into jraphx's lin_qkv."""
    wq = _np(theirs.lin_query.weight).T
    wk = _np(theirs.lin_key.weight).T
    wv = _np(theirs.lin_value.weight).T
    ours.lin_qkv.kernel[...] = jnp.asarray(np.concatenate([wq, wk, wv], axis=1))
    bq = _np(theirs.lin_query.bias)
    bk = _np(theirs.lin_key.bias)
    bv = _np(theirs.lin_value.bias)
    ours.lin_qkv.bias[...] = jnp.asarray(np.concatenate([bq, bk, bv]))
    if theirs.lin_skip is not None and ours.lin_skip is not None:
        _copy_linear(ours.lin_skip, theirs.lin_skip)
    if getattr(theirs, "lin_edge", None) is not None and ours.lin_edge is not None:
        _copy_linear(ours.lin_edge, theirs.lin_edge)
    if getattr(theirs, "lin_beta", None) is not None and ours.lin_beta is not None:
        _copy_linear(ours.lin_beta, theirs.lin_beta)


@pytest.mark.parametrize("concat", [True, False])
def test_transformer_conv_matches_pyg(concat):
    x, edge_index, _ = _graph(in_features=5)

    theirs = pyg_nn.TransformerConv(5, 4, heads=3, concat=concat)
    ours = TransformerConv(5, 4, heads=3, concat=concat, rngs=nnx.Rngs(0))
    _copy_transformer(ours, theirs, heads=3, out_features=4)

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )


def test_transformer_conv_matches_pyg_with_edge_features():
    x, edge_index, edge_attr = _graph(in_features=5, edge_dim=3)

    theirs = pyg_nn.TransformerConv(5, 4, heads=2, edge_dim=3)
    ours = TransformerConv(5, 4, heads=2, edge_dim=3, rngs=nnx.Rngs(0))
    _copy_transformer(ours, theirs, heads=2, out_features=4)

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index), jnp.asarray(edge_attr)),
        theirs(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attr)),
    )


def test_transformer_conv_matches_pyg_with_beta():
    x, edge_index, _ = _graph(in_features=5)

    theirs = pyg_nn.TransformerConv(5, 4, heads=2, beta=True)
    ours = TransformerConv(5, 4, heads=2, beta=True, rngs=nnx.Rngs(0))
    _copy_transformer(ours, theirs, heads=2, out_features=4)

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(edge_index)),
        theirs(torch.tensor(x), torch.tensor(edge_index)),
    )
