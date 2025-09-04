"""Test JumpingKnowledge module for JraphX.

Converted from PyTorch Geometric test_jumping_knowledge.py to test JraphX functionality.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from jraphx.nn.models.jumping_knowledge import JumpingKnowledge


def test_jumping_knowledge():
    """Test JumpingKnowledge aggregation with different modes."""
    num_nodes, channels, num_layers = 100, 17, 5

    # Create random key for reproducible test data
    key = nnx.Rngs(42)

    # Create layer representations
    xs = []
    for _ in range(num_layers):
        x = jnp.array(nnx.initializers.normal()(key.params(), (num_nodes, channels)))
        xs.append(x)

    # Test concatenation mode
    model = JumpingKnowledge("cat")

    output = model(xs)
    assert output.shape == (num_nodes, channels * num_layers)

    # Test max pooling mode
    model = JumpingKnowledge("max")

    output = model(xs)
    assert output.shape == (num_nodes, channels)

    # Test LSTM mode
    model = JumpingKnowledge("lstm", channels, num_layers, rngs=nnx.Rngs(42))

    output = model(xs)
    assert output.shape == (num_nodes, channels)


def test_jumping_knowledge_different_sizes():
    """Test JumpingKnowledge with different input sizes."""
    # Test with smaller dimensions
    num_nodes, channels, num_layers = 10, 8, 3

    key = nnx.Rngs(42)
    xs = []
    for _ in range(num_layers):
        x = jnp.array(nnx.initializers.normal()(key.params(), (num_nodes, channels)))
        xs.append(x)

    for mode in ["cat", "max", "lstm"]:
        if mode == "lstm":
            model = JumpingKnowledge(mode, channels, num_layers, rngs=nnx.Rngs(42))
        else:
            model = JumpingKnowledge(mode)

        output = model(xs)

        if mode == "cat":
            expected_shape = (num_nodes, channels * num_layers)
        else:  # max or lstm
            expected_shape = (num_nodes, channels)

        assert output.shape == expected_shape


def test_jumping_knowledge_single_layer():
    """Test JumpingKnowledge with single layer input."""
    num_nodes, channels = 50, 12

    key = nnx.Rngs(42)
    x = jnp.array(nnx.initializers.normal()(key.params(), (num_nodes, channels)))
    xs = [x]  # Single layer

    # Test concatenation (should return the single input)
    model = JumpingKnowledge("cat")
    output = model(xs)
    assert output.shape == (num_nodes, channels)
    assert jnp.allclose(output, x)

    # Test max (should return the single input)
    model = JumpingKnowledge("max")
    output = model(xs)
    assert output.shape == (num_nodes, channels)
    assert jnp.allclose(output, x)

    # Test LSTM with single layer
    model = JumpingKnowledge("lstm", channels, 1, rngs=nnx.Rngs(42))
    output = model(xs)
    assert output.shape == (num_nodes, channels)


def test_jumping_knowledge_modes():
    """Test that different modes produce different outputs."""
    num_nodes, channels, num_layers = 20, 6, 4

    xs = []
    for i in range(num_layers):
        # Create different patterns for each layer
        x = jnp.ones((num_nodes, channels)) * (i + 1)
        xs.append(x)

    # Test concatenation - should stack all layers
    cat_model = JumpingKnowledge("cat")
    cat_output = cat_model(xs)
    assert cat_output.shape == (num_nodes, channels * num_layers)

    # Test max - should select maximum values
    max_model = JumpingKnowledge("max")
    max_output = max_model(xs)
    assert max_output.shape == (num_nodes, channels)
    # Should select the last layer since it has the highest values
    assert jnp.allclose(max_output, jnp.ones((num_nodes, channels)) * num_layers)


def test_invalid_mode():
    """Test that invalid modes raise appropriate errors."""
    with pytest.raises(AssertionError):
        JumpingKnowledge("invalid_mode")


def test_lstm_mode_requirements():
    """Test that LSTM mode requires proper parameters."""
    # Should fail without num_features
    with pytest.raises(AssertionError):
        JumpingKnowledge("lstm")

    # Should fail without num_layers
    with pytest.raises(AssertionError):
        JumpingKnowledge("lstm", num_features=10)

    # Should work with both parameters
    model = JumpingKnowledge("lstm", num_features=10, num_layers=3, rngs=nnx.Rngs(42))
    assert model.mode == "lstm"
    assert model.features == 10
    assert model.num_layers == 3


# TODO: PyG-specific features not directly convertible:
# - JIT scripting tests (JAX has different compilation)
# - HeteroJumpingKnowledge (heterogeneous graph support not in JraphX)


def test_empty_input():
    """Test behavior with empty layer list."""
    model = JumpingKnowledge("cat")

    # Empty list should raise error or handle gracefully
    with pytest.raises((IndexError, ValueError)):
        model([])


def test_mismatched_shapes():
    """Test behavior when layer shapes don't match."""
    model = JumpingKnowledge("max")

    # Different number of nodes should fail
    xs = [jnp.ones((10, 5)), jnp.ones((20, 5))]  # Different number of nodes

    with pytest.raises((ValueError, IndexError)):
        model(xs)

    # Different number of features should fail
    xs = [jnp.ones((10, 5)), jnp.ones((10, 8))]  # Different number of features

    with pytest.raises((ValueError, IndexError)):
        model(xs)


if __name__ == "__main__":
    # Run basic tests
    test_jumping_knowledge()
    test_jumping_knowledge_modes()
    print("JumpingKnowledge tests passed!")
