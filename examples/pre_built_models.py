"""Example demonstrating pre-built GNN models in JraphX.

This example shows:
1. Using pre-built GCN, GAT, GraphSAGE, and GIN models
2. Different configurations (normalization, dropout, JumpingKnowledge)
3. Node classification with the Karate Club dataset
4. Comparison of different architectures
"""

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"


import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from jraphx import Data  # noqa: E402
from jraphx.nn.models import GAT, GCN, GIN, GraphSAGE  # noqa: E402


def create_karate_club_data():
    """Create the Zachary's Karate Club graph.

    Edge list from the original Zachary's karate club network.
    Reference: https://en.wikipedia.org/wiki/Zachary%27s_karate_club
    """
    # fmt: off
    # Edge list for the Karate Club graph (78 edges, using 0-based indexing)
    edges = [
        [0,  1], [0,  2], [0,  3], [0,  4], [0,  5], [0,  6], [0,  7], [0,  8],
        [0, 10], [0, 11], [0, 12], [0, 13], [0, 17], [0, 19], [0, 21], [0, 31],
        [1,  2], [1,  3], [1,  7], [1, 13], [1, 17], [1, 19], [1, 21], [1, 30],
        [2,  3], [2,  7], [2,  8], [2,  9], [2, 13], [2, 27], [2, 28], [2, 32],
        [3,  7], [3, 12], [3, 13],
        [4,  6], [4, 10],
        [5,  6], [5, 10], [5, 16],
        [6, 16],
        [8, 30], [8, 32], [8, 33],
        [9, 33],
        [13, 33],
        [14, 32], [14, 33],
        [15, 32], [15, 33],
        [18, 32], [18, 33],
        [19, 33],
        [20, 32], [20, 33],
        [22, 32], [22, 33],
        [23, 25], [23, 27], [23, 29], [23, 32], [23, 33],
        [24, 25], [24, 27], [24, 31],
        [25, 31],
        [26, 29], [26, 33],
        [27, 33],
        [28, 31], [28, 33],
        [29, 32], [29, 33],
        [30, 32], [30, 33],
        [31, 32], [31, 33],
        [32, 33],
    ]
    # fmt: on

    edge_index = jnp.array(edges).T
    # Make bidirectional
    edge_index = jnp.concatenate([edge_index, jnp.array([edge_index[1], edge_index[0]])], axis=1)

    # Node features (random for demonstration, in practice could be node degrees, etc.)
    num_nodes = 34
    x = jnp.array(np.random.randn(num_nodes, 16))

    # Labels (community assignments - simplified to 2 communities)
    y = jnp.array([0] * 17 + [1] * 17)

    # Train/val/test masks
    train_mask = jnp.zeros(num_nodes, dtype=bool)
    train_mask = train_mask.at[0].set(True)
    train_mask = train_mask.at[33].set(True)
    train_mask = train_mask.at[1].set(True)
    train_mask = train_mask.at[32].set(True)

    return Data(x=x, edge_index=edge_index, y=y), train_mask


def train_model(model, data, train_mask, num_epochs=200, lr=0.01):
    """Train a GNN model on node classification."""
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    def train_step(model, optimizer, data, mask):
        @nnx.value_and_grad
        def loss_fn(model, data, mask):
            logits = model(data.x, data.edge_index)
            # Only compute loss on training nodes
            train_logits = logits[mask]
            train_labels = data.y[mask]
            loss = optax.softmax_cross_entropy_with_integer_labels(train_logits, train_labels)
            return loss.mean()

        loss, grads = loss_fn(model, data, mask)
        optimizer.update(model, grads)
        return loss

    def evaluate(model, data):
        logits = model(data.x, data.edge_index)
        pred = jnp.argmax(logits, axis=-1)
        acc = (pred == data.y).mean()
        return acc

    # Training loop
    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, data, train_mask)

        if epoch % 50 == 0:
            acc = evaluate(model, data)
            print(f"  Epoch {epoch:3d}, Loss: {loss:.4f}, Accuracy: {acc:.3f}")

    # Final evaluation
    acc = evaluate(model, data)
    return acc


def demo_basic_models():
    """Demonstrate basic usage of pre-built models."""
    print("=" * 60)
    print("Basic Pre-built Models Demo")
    print("=" * 60)

    data, train_mask = create_karate_club_data()

    # 1. GCN Model
    print("\n1. GCN Model:")
    rngs = nnx.Rngs(0)
    gcn = GCN(
        in_features=16,
        hidden_features=32,
        num_layers=2,
        out_features=2,
        dropout_rate=0.5,
        rngs=rngs,
    )
    acc = train_model(gcn, data, train_mask, num_epochs=100)
    print(f"  Final GCN Accuracy: {acc:.3f}")

    # 2. GAT Model
    print("\n2. GAT Model:")
    rngs = nnx.Rngs(0)
    gat = GAT(
        in_features=16,
        hidden_features=32,
        num_layers=2,
        out_features=2,
        heads=4,
        dropout_rate=0.5,
        rngs=rngs,
    )
    acc = train_model(gat, data, train_mask, num_epochs=100)
    print(f"  Final GAT Accuracy: {acc:.3f}")

    # 3. GraphSAGE Model
    print("\n3. GraphSAGE Model:")
    rngs = nnx.Rngs(0)
    sage = GraphSAGE(
        in_features=16,
        hidden_features=32,
        num_layers=2,
        out_features=2,
        aggr="mean",
        dropout_rate=0.5,
        rngs=rngs,
    )
    acc = train_model(sage, data, train_mask, num_epochs=100)
    print(f"  Final GraphSAGE Accuracy: {acc:.3f}")

    # 4. GIN Model
    print("\n4. GIN Model:")
    rngs = nnx.Rngs(0)
    gin = GIN(
        in_features=16,
        hidden_features=32,
        num_layers=2,
        out_features=2,
        dropout_rate=0.5,
        train_eps=True,
        rngs=rngs,
    )
    acc = train_model(gin, data, train_mask, num_epochs=100)
    print(f"  Final GIN Accuracy: {acc:.3f}")


def demo_advanced_configurations():
    """Demonstrate advanced model configurations."""
    print("\n" + "=" * 60)
    print("Advanced Configurations Demo")
    print("=" * 60)

    data, train_mask = create_karate_club_data()

    # 1. Model with Normalization
    print("\n1. GCN with Layer Normalization:")
    rngs = nnx.Rngs(0)
    gcn_norm = GCN(
        in_features=16,
        hidden_features=32,
        num_layers=3,
        out_features=2,
        norm="layer_norm",
        dropout_rate=0.3,
        rngs=rngs,
    )
    acc = train_model(gcn_norm, data, train_mask, num_epochs=100)
    print(f"  Final Accuracy: {acc:.3f}")

    # 2. Model with JumpingKnowledge
    print("\n2. GAT with JumpingKnowledge (max):")
    rngs = nnx.Rngs(0)
    gat_jk = GAT(
        in_features=16,
        hidden_features=32,
        num_layers=3,
        out_features=2,
        heads=2,
        jk="max",
        rngs=rngs,
    )
    acc = train_model(gat_jk, data, train_mask, num_epochs=100)
    print(f"  Final Accuracy: {acc:.3f}")

    # 3. Model with Residual Connections
    print("\n3. GCN with Residual Connections:")
    rngs = nnx.Rngs(0)
    gcn_res = GCN(
        in_features=16,
        hidden_features=32,
        num_layers=4,
        out_features=2,
        residual=True,
        dropout_rate=0.2,
        rngs=rngs,
    )
    acc = train_model(gcn_res, data, train_mask, num_epochs=100)
    print(f"  Final Accuracy: {acc:.3f}")

    # 4. Deep Model with Multiple Features
    print("\n4. Deep GIN with Normalization and JumpingKnowledge:")
    rngs = nnx.Rngs(0)
    gin_deep = GIN(
        in_features=16,
        hidden_features=64,
        num_layers=5,
        out_features=2,
        norm="layer_norm",
        jk="cat",
        residual=True,
        dropout_rate=0.3,
        rngs=rngs,
    )
    acc = train_model(gin_deep, data, train_mask, num_epochs=100, lr=0.005)
    print(f"  Final Accuracy: {acc:.3f}")


def demo_model_comparison():
    """Compare different models on the same task."""
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    data, train_mask = create_karate_club_data()

    models = {
        "GCN": lambda rngs: GCN(16, 32, 2, 2, dropout_rate=0.5, rngs=rngs),
        "GAT": lambda rngs: GAT(16, 32, 2, 2, heads=4, dropout_rate=0.5, rngs=rngs),
        "GraphSAGE": lambda rngs: GraphSAGE(16, 32, 2, 2, dropout_rate=0.5, rngs=rngs),
        "GIN": lambda rngs: GIN(16, 32, 2, 2, dropout_rate=0.5, rngs=rngs),
    }

    results = {}
    for name, model_fn in models.items():
        print(f"\nTraining {name}...")
        rngs = nnx.Rngs(42)  # Same seed for fair comparison
        model = model_fn(rngs)
        acc = train_model(model, data, train_mask, num_epochs=150)
        results[name] = acc

    print("\n" + "-" * 40)
    print("Final Results:")
    for name, acc in results.items():
        print(f"  {name:12s}: {acc:.3f}")

    best_model = max(results, key=results.get)
    print(f"\nBest model: {best_model} with accuracy {results[best_model]:.3f}")


if __name__ == "__main__":
    print("JraphX Pre-built Models Example")
    print("=" * 60)

    # Set JAX to use CPU
    jax.config.update("jax_platform_name", "cpu")

    # Run demos
    demo_basic_models()
    demo_advanced_configurations()
    demo_model_comparison()

    print("\n" + "=" * 60)
    print("✅ Pre-built models example completed!")
