"""Numerical parity of jraphx normalization layers against torch_geometric."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

torch = pytest.importorskip("torch")
pyg_norm = pytest.importorskip("torch_geometric.nn.norm")

from jraphx.nn.norm import BatchNorm, GraphNorm, LayerNorm  # noqa: E402


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _assert_close(jax_result, torch_result, atol=1e-5):
    np.testing.assert_allclose(_np(jax_result), _np(torch_result), atol=atol, rtol=1e-5)


@pytest.fixture
def node_features():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(9, 4)).astype(np.float32) * 3.0 + 1.0
    batch = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    return x, batch


def test_batch_norm_training_matches_pyg(node_features):
    x, _ = node_features
    # PyG's torch momentum is the batch weight; jraphx's is the decay, so
    # torch momentum=0.1 corresponds to jraphx momentum=0.9.
    theirs = pyg_norm.BatchNorm(4, momentum=0.1)
    theirs.train()
    ours = BatchNorm(4, momentum=0.9, rngs=nnx.Rngs(0))
    ours.train()

    for _ in range(3):
        out_theirs = theirs(torch.tensor(x))
        out_ours = ours(jnp.asarray(x))

    _assert_close(out_ours, out_theirs)
    _assert_close(ours.running_mean[...], theirs.module.running_mean)
    _assert_close(ours.running_var[...], theirs.module.running_var)

    # Evaluation uses the running statistics
    theirs.eval()
    ours.eval()
    _assert_close(ours(jnp.asarray(x)), theirs(torch.tensor(x)))


def test_layer_norm_node_mode_matches_pyg(node_features):
    x, _ = node_features
    theirs = pyg_norm.LayerNorm(4, mode="node")
    ours = LayerNorm(4, mode="node", rngs=nnx.Rngs(0))
    _assert_close(ours(jnp.asarray(x)), theirs(torch.tensor(x)))


def test_layer_norm_graph_mode_matches_pyg(node_features):
    x, batch = node_features
    theirs = pyg_norm.LayerNorm(4, mode="graph")
    ours = LayerNorm(4, mode="graph", rngs=nnx.Rngs(0))
    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(batch), batch_size=2),
        theirs(torch.tensor(x), torch.tensor(batch)),
    )


def test_graph_norm_matches_pyg(node_features):
    x, batch = node_features
    theirs = pyg_norm.GraphNorm(4)
    with torch.no_grad():
        theirs.weight.copy_(torch.tensor([1.5, 0.5, 2.0, 1.0]))
        theirs.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        theirs.mean_scale.copy_(torch.tensor([0.9, 1.1, 1.0, 0.7]))

    ours = GraphNorm(4)
    ours.weight[...] = jnp.array([1.5, 0.5, 2.0, 1.0])
    ours.bias[...] = jnp.array([0.1, -0.2, 0.3, 0.0])
    ours.mean_scale[...] = jnp.array([0.9, 1.1, 1.0, 0.7])

    _assert_close(
        ours(jnp.asarray(x), jnp.asarray(batch), batch_size=2),
        theirs(torch.tensor(x), torch.tensor(batch)),
    )
