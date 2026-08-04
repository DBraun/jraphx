"""Test MLP module for JraphX.

Converted from PyTorch Geometric test_mlp.py to test JraphX functionality.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from jraphx.nn.models.mlp import MLP


@pytest.mark.parametrize("norm", ["batch_norm", None])
@pytest.mark.parametrize("act_first", [False, True])
@pytest.mark.parametrize("plain_last", [False, True])
def test_mlp(norm, act_first, plain_last):
    """Test MLP with different configurations."""
    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
            ],
            [
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
            ],
            [
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
            ],
        ]
    )

    # Test feature list specification
    mlp = MLP(
        [16, 32, 32, 64],
        norm=norm,
        act_first=act_first,
        plain_last=plain_last,
        rngs=nnx.Rngs(12345),
    )

    output = mlp(x)
    assert output.shape == (4, 64)

    # Test alternative specification with explicit parameters
    mlp2 = MLP(
        in_features=16,
        hidden_features=32,
        out_features=64,
        num_layers=3,
        norm=norm,
        act_first=act_first,
        plain_last=plain_last,
        rngs=nnx.Rngs(12345),
    )

    output2 = mlp2(x)
    assert output2.shape == (4, 64)


@pytest.mark.parametrize(
    "norm",
    [
        "batch_norm",
        "layer_norm",
    ],
)
def test_batch(norm):
    """Test MLP with batch processing."""
    x = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    batch = jnp.array([0, 0, 1])

    model = MLP(
        in_features=8,
        hidden_features=16,
        out_features=32,
        num_layers=2,
        norm=norm,
        rngs=nnx.Rngs(42),
    )

    output = model(x, batch=batch)
    assert output.shape == (3, 32)


def test_mlp_properties():
    """Test MLP properties and different configurations."""
    # Test single layer
    mlp = MLP([16, 32], rngs=nnx.Rngs(42))
    assert mlp.in_features == 16
    assert mlp.out_features == 32
    assert mlp.num_layers == 1

    # Test multiple layers
    mlp = MLP(in_features=16, hidden_features=32, out_features=64, num_layers=4, rngs=nnx.Rngs(42))
    assert mlp.in_features == 16
    assert mlp.out_features == 64
    assert mlp.num_layers == 4
    assert mlp.feature_list == [16, 32, 32, 32, 64]


def test_mlp_activation_functions():
    """Test MLP with different activation functions."""
    x = jnp.ones((4, 16))

    for act in [None, nnx.relu, nnx.tanh, nnx.gelu]:
        mlp = MLP([16, 32, 64], act=act, rngs=nnx.Rngs(42))

        output = mlp(x)
        assert output.shape == (4, 64)


def test_mlp_dropout():
    """Test MLP with dropout."""
    x = jnp.ones((4, 16))

    mlp = MLP([16, 32, 64], dropout_rate=0.5, rngs=nnx.Rngs(42))

    output = mlp(x)
    assert output.shape == (4, 64)


def test_mlp_bias():
    """Test MLP with and without bias."""
    x = jnp.ones((4, 16))

    for bias in [True, False]:
        mlp = MLP([16, 32, 64], bias=bias, rngs=nnx.Rngs(42))

        output = mlp(x)
        assert output.shape == (4, 64)


def test_mlp_error_cases():
    """Test MLP error handling."""
    # Should fail without feature_list or in_features
    with pytest.raises(ValueError):
        MLP()

    # Should fail without num_layers when using in_features
    with pytest.raises(ValueError):
        MLP(in_features=16)

    # Should fail without hidden_features for multi-layer network
    with pytest.raises(ValueError):
        MLP(in_features=16, num_layers=3, out_features=32)

    # Should fail without out_features
    with pytest.raises(ValueError):
        MLP(in_features=16, num_layers=2, hidden_features=32)


def test_mlp_unknown_norm_raises():
    """An unrecognized normalization name is rejected instead of silently ignored."""
    with pytest.raises(ValueError, match="Unknown normalization"):
        MLP([16, 32, 64], norm="layernorm", rngs=nnx.Rngs(42))

    # GraphNorm is not supported by MLP, which never receives a batch vector
    with pytest.raises(ValueError, match="Unknown normalization"):
        MLP([16, 32, 64], norm="graph_norm", rngs=nnx.Rngs(42))

    # The check fires even when no normalization layer would be created
    with pytest.raises(ValueError, match="Unknown normalization"):
        MLP([16, 32], norm="graph_norm", rngs=nnx.Rngs(42))


def test_mlp_act_none_disables_activation():
    """``act=None`` makes the MLP a plain composition of linear layers."""
    x = jnp.arange(8.0).reshape(2, 4)

    mlp = MLP([4, 4, 4], act=None, plain_last=False, rngs=nnx.Rngs(42))

    expected = mlp.lins[1](mlp.lins[0](x))
    assert jnp.allclose(mlp(x), expected, atol=1e-6)

    relu_mlp = MLP([4, 4, 4], act=nnx.relu, plain_last=False, rngs=nnx.Rngs(42))
    assert not jnp.allclose(relu_mlp(x), expected, atol=1e-6)


def test_mlp_single_layer():
    """Test MLP with single layer."""
    x = jnp.ones((4, 16))

    mlp = MLP(in_features=16, out_features=32, num_layers=1, rngs=nnx.Rngs(42))

    output = mlp(x)
    assert output.shape == (4, 32)
    assert mlp.num_layers == 1


def test_mlp_large_network():
    """Test MLP with many layers."""
    x = jnp.ones((2, 8))

    mlp = MLP(in_features=8, hidden_features=16, out_features=4, num_layers=10, rngs=nnx.Rngs(42))

    output = mlp(x)
    assert output.shape == (2, 4)
    assert mlp.num_layers == 10


# TODO: PyG-specific features not directly convertible:
# - JIT scripting tests (JAX has different compilation)
# - return_emb parameter (not implemented in JraphX MLP)
# - Fine-grained control over dropout/bias per layer (not implemented)
# - Different normalization types per layer


def test_mlp_different_sizes():
    """Test MLP with various input/output sizes."""
    test_cases = [
        ([8, 16], (3, 16)),  # Minimum 2 layers required
        ([10, 20, 30], (5, 30)),
        ([100, 50, 25, 10], (1, 10)),
    ]

    for feature_list, expected_output_shape in test_cases:
        input_shape = (expected_output_shape[0], feature_list[0])
        x = jnp.ones(input_shape)

        mlp = MLP(feature_list, rngs=nnx.Rngs(42))
        output = mlp(x)
        assert output.shape == expected_output_shape


def test_mlp_normalization_types():
    """Test different normalization types."""
    x = jnp.ones((4, 16))

    for norm in [None, "batch_norm", "layer_norm"]:
        mlp = MLP([16, 32, 64], norm=norm, rngs=nnx.Rngs(42))

        output = mlp(x)
        assert output.shape == (4, 64)


if __name__ == "__main__":
    # Run basic tests
    test_mlp(None, False, True)
    test_mlp_properties()
    print("MLP tests passed!")
