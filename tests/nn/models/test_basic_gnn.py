"""Test basic GNN models for JraphX.

Converted from PyTorch Geometric test_basic_gnn.py to test JraphX functionality.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from jraphx.nn.models import GAT, GCN, GIN, GraphSAGE

# Test parameters - matching PyG test structure
out_dims = [None, 8]
dropouts = [0.0, 0.5]
acts = [None, nnx.relu, nnx.leaky_relu]  # JAX activation functions
norms = [None, "batch_norm", "layer_norm"]
jks = [None, "last", "cat", "max", "lstm"]


def create_test_data():
    """Create test graph data."""
    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    edge_index = jnp.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    return x, edge_index


@pytest.mark.parametrize("out_dim", out_dims)
@pytest.mark.parametrize("dropout", dropouts)
@pytest.mark.parametrize("act", acts)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("jk", jks)
def test_gcn(out_dim, dropout, act, norm, jk):
    """Test GCN model with various configurations."""
    x, edge_index = create_test_data()
    out_features = 16 if out_dim is None else out_dim

    model = GCN(
        in_features=8,
        hidden_features=16,
        num_layers=3,
        out_features=out_dim,
        dropout_rate=dropout,
        act=act,
        norm=norm,
        jk=jk,
        rngs=nnx.Rngs(42),
    )

    # Test forward pass
    output = model(x, edge_index)
    assert output.shape == (3, out_features)


@pytest.mark.parametrize("out_dim", out_dims)
@pytest.mark.parametrize("dropout", dropouts)
@pytest.mark.parametrize("act", acts)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("jk", jks)
def test_graph_sage(out_dim, dropout, act, norm, jk):
    """Test GraphSAGE model with various configurations."""
    x, edge_index = create_test_data()
    out_features = 16 if out_dim is None else out_dim

    model = GraphSAGE(
        in_features=8,
        hidden_features=16,
        num_layers=3,
        out_features=out_dim,
        dropout_rate=dropout,
        act=act,
        norm=norm,
        jk=jk,
        rngs=nnx.Rngs(42),
    )

    # Test forward pass
    output = model(x, edge_index)
    assert output.shape == (3, out_features)


@pytest.mark.parametrize("out_dim", out_dims)
@pytest.mark.parametrize("dropout", dropouts)
@pytest.mark.parametrize("act", acts)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("jk", jks)
def test_gin(out_dim, dropout, act, norm, jk):
    """Test GIN model with various configurations."""
    x, edge_index = create_test_data()
    out_features = 16 if out_dim is None else out_dim

    model = GIN(
        in_features=8,
        hidden_features=16,
        num_layers=3,
        out_features=out_dim,
        dropout_rate=dropout,
        act=act,
        norm=norm,
        jk=jk,
        rngs=nnx.Rngs(42),
    )

    # Test forward pass
    output = model(x, edge_index)
    assert output.shape == (3, out_features)


@pytest.mark.parametrize("out_dim", out_dims)
@pytest.mark.parametrize("dropout", dropouts)
@pytest.mark.parametrize("act", acts)
@pytest.mark.parametrize("norm", norms)
@pytest.mark.parametrize("jk", jks)
def test_gat(out_dim, dropout, act, norm, jk):
    """Test GAT model with various configurations."""
    x, edge_index = create_test_data()
    out_features = 16 if out_dim is None else out_dim

    for v2 in [False, True]:
        model = GAT(
            in_features=8,
            hidden_features=16,
            num_layers=3,
            out_features=out_dim,
            v2=v2,
            dropout_rate=dropout,
            act=act,
            norm=norm,
            jk=jk,
            rngs=nnx.Rngs(42),
        )

        # Test forward pass
        output = model(x, edge_index)
        assert output.shape == (3, out_features)

        # Test with multiple heads
        model = GAT(
            in_features=8,
            hidden_features=16,
            num_layers=3,
            out_features=out_dim,
            v2=v2,
            dropout_rate=dropout,
            act=act,
            norm=norm,
            jk=jk,
            heads=4,
            rngs=nnx.Rngs(42),
        )

        output = model(x, edge_index)
        assert output.shape == (3, out_features)


@pytest.mark.parametrize("out_dim", out_dims)
@pytest.mark.parametrize("jk", jks)
@pytest.mark.parametrize("model_cls", [GCN, GraphSAGE, GIN, GAT])
def test_one_layer_gnn(out_dim, jk, model_cls):
    """A single-layer model honours out_features exactly like a deeper one."""
    x, edge_index = create_test_data()
    out_features = 16 if out_dim is None else out_dim

    kwargs = {"heads": 4} if model_cls is GAT else {}
    model = model_cls(
        in_features=8,
        hidden_features=16,
        num_layers=1,
        out_features=out_dim,
        jk=jk,
        rngs=nnx.Rngs(42),
        **kwargs,
    )

    output = model(x, edge_index)
    assert output.shape == (3, out_features)
    # The advertised width and the emitted width must not drift apart
    assert model.out_features == output.shape[-1]


def test_batch_processing():
    """Test batch processing with batch vectors."""
    x, edge_index = create_test_data()
    batch = jnp.array([0, 0, 1])

    # Test with batch-compatible normalization
    for norm in ["layer_norm", "graph_norm"]:
        model = GraphSAGE(
            in_features=8, hidden_features=16, num_layers=2, norm=norm, rngs=nnx.Rngs(42)
        )

        output = model(x, edge_index, batch=batch)
        assert output.shape == (3, 16)


# TODO: PyG-specific tests that can't be directly converted:
# - test_jit() - JAX uses different JIT compilation
# - test_basic_gnn_inference() - requires PyG neighbor sampling
# - test_compile_basic() - JAX compilation works differently
# - test_packaging() - PyG-specific model serialization
# - test_onnx() - ONNX export not supported in JraphX
# - test_trim_to_layer() - PyG-specific feature
# - test_compile_graph_breaks() - PyG-specific compilation checks
# - test_basic_gnn_cache() - PyG-specific caching mechanism

# TODO: PyG models not yet implemented in JraphX:
# - PNA (Principal Neighbourhood Aggregation)
# - EdgeCNN (EdgeConv)


def test_residual_connections():
    """Test residual connections work properly."""
    x, edge_index = create_test_data()

    model = GraphSAGE(
        in_features=8,
        hidden_features=8,  # Same size for residual connections
        num_layers=3,
        residual=True,
        rngs=nnx.Rngs(42),
    )

    output = model(x, edge_index)
    assert output.shape == (3, 8)


def test_unknown_norm_raises():
    """An unrecognized normalization name is rejected instead of silently ignored."""
    with pytest.raises(ValueError, match="Unknown normalization"):
        GCN(
            in_features=8,
            hidden_features=16,
            num_layers=2,
            out_features=4,
            norm="layernorm",
            rngs=nnx.Rngs(42),
        )


def test_act_none_disables_activation():
    """``act=None`` composes the convolutions without any non-linearity."""
    x, edge_index = create_test_data()

    model = GCN(
        in_features=8,
        hidden_features=8,
        num_layers=2,
        out_features=8,
        act=None,
        rngs=nnx.Rngs(42),
    )

    expected = model.convs[1](model.convs[0](x, edge_index), edge_index)
    assert jnp.allclose(model(x, edge_index), expected, atol=1e-6)

    relu_model = GCN(
        in_features=8,
        hidden_features=8,
        num_layers=2,
        out_features=8,
        act=nnx.relu,
        rngs=nnx.Rngs(42),
    )
    assert not jnp.allclose(relu_model(x, edge_index), expected, atol=1e-6)


def test_residual_applies_to_first_layer():
    """The residual connection is added whenever the widths line up."""
    x, edge_index = create_test_data()

    model = GraphSAGE(
        in_features=8,
        hidden_features=8,
        num_layers=1,
        out_features=8,
        residual=True,
        rngs=nnx.Rngs(42),
    )

    expected = model.convs[0](x, edge_index) + x
    assert jnp.allclose(model(x, edge_index), expected, atol=1e-6)


def test_edge_weight_is_not_passed_as_edge_attr():
    """Edge weights only reach convolutions that support them."""
    x, edge_index = create_test_data()
    edge_weight = jnp.array([0.5, 1.5, 2.5, 3.5])

    # GAT consumes edge attributes, not edge weights: a one-dimensional
    # edge_weight must not be reinterpreted as an edge feature
    gat = GAT(
        in_features=8,
        hidden_features=16,
        num_layers=2,
        out_features=4,
        edge_dim=1,
        rngs=nnx.Rngs(42),
    )
    assert jnp.allclose(gat(x, edge_index, edge_weight=edge_weight), gat(x, edge_index), atol=1e-6)

    # ... while a genuine edge attribute does change the GAT output
    edge_attr = edge_weight.reshape(-1, 1)
    assert not jnp.allclose(gat(x, edge_index, edge_attr=edge_attr), gat(x, edge_index), atol=1e-6)

    # GCN does consume edge weights
    gcn = GCN(
        in_features=8,
        hidden_features=16,
        num_layers=2,
        out_features=4,
        rngs=nnx.Rngs(42),
    )
    assert not jnp.allclose(
        gcn(x, edge_index, edge_weight=edge_weight), gcn(x, edge_index), atol=1e-6
    )


def test_gin_inner_mlp_configuration():
    """The MLP inside GINConv drops no activations twice and never uses GraphNorm."""
    model = GIN(
        in_features=8,
        hidden_features=16,
        num_layers=2,
        out_features=4,
        dropout_rate=0.5,
        norm="graph_norm",
        rngs=nnx.Rngs(42),
    )

    # Dropout is applied once per block, by the model and not by the inner MLP
    assert model.dropout is not None
    assert model.convs[0].nn.dropout is None

    # GraphNorm is applied between blocks; the inner MLP normalizes per node
    assert type(model.norms[0]).__name__ == "GraphNorm"
    assert type(model.convs[0].nn.norms[0]).__name__ == "LayerNorm"


def test_different_output_features():
    """Test models with different output feature sizes."""
    x, edge_index = create_test_data()

    for out_features in [4, 16, 32]:
        model = GCN(
            in_features=8,
            hidden_features=16,
            num_layers=2,
            out_features=out_features,
            rngs=nnx.Rngs(42),
        )

        output = model(x, edge_index)
        assert output.shape == (3, out_features)


if __name__ == "__main__":
    # Run a basic test
    test_gcn(None, 0.0, None, None, None)
    print("Basic GNN tests passed!")
