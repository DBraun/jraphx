"""
Graph Convolutional Network Example using JraphX
================================================

This example demonstrates using jraphx's built-in GCNConv layer instead of
implementing GCN from scratch. It shows how to:
1. Use jraphx's GCNConv layer for message passing
2. Work with jraphx's Data structure
3. Train a GCN model on synthetic graph data
4. Use data parallelism with nnx.shard_map for multi-device training

This is the jraphx version of gcn_standalone.py
"""

import os

# Pretend to have multiple devices:
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["JAX_PLATFORMS"] = "cpu"


import jax  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

# Import jraphx components
from jraphx import Data  # noqa: E402
from jraphx.nn.conv import GCNConv  # noqa: E402


class SimpleGCN(nnx.Module):
    """
    A simple 2-layer GCN for node classification using jraphx.

    Architecture:
        Input -> GCNConv(hidden_dim) -> ReLU -> Dropout -> GCNConv(out_dim) -> Output
    """

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, *, rngs: nnx.Rngs
    ):
        self.conv1 = GCNConv(
            in_features, hidden_features, normalize=True, add_self_loops=True, bias=True, rngs=rngs
        )
        self.conv2 = GCNConv(
            hidden_features, out_features, normalize=True, add_self_loops=True, bias=True, rngs=rngs
        )
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = nnx.relu(x)
        x = self.dropout(x)

        # Second GCN layer (no activation for final layer)
        x = self.conv2(x, edge_index)
        return x


def create_synthetic_graph_data(rngs: nnx.Rngs, num_nodes: int = 20, num_edges: int = 40):
    """
    Create synthetic graph data using jraphx's Data structure.

    Returns:
        data: jraphx Data object containing the graph
    """
    # Random node features (32-dimensional)
    x = rngs.normal((num_nodes, 32))

    # Random edges - sample pairs of nodes
    edge_index = rngs.choice(num_nodes, shape=(2, num_edges), replace=True)

    # Random labels (4 classes)
    y = rngs.randint((num_nodes,), 0, 4)

    # Create jraphx Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def batch_graphs_jraphx(graphs_list):
    """
    Batch multiple jraphx Data objects into a single graph.

    Args:
        graphs_list: List of jraphx Data objects

    Returns:
        Batched graph data and batch assignment vector
    """
    # Stack all node features
    x_batch = jnp.vstack([g.x for g in graphs_list])
    y_batch = jnp.hstack([g.y for g in graphs_list])

    # Offset edge indices for each graph
    edge_batch_list = []
    node_offset = 0
    batch_vector = []

    for i, graph in enumerate(graphs_list):
        num_nodes = graph.num_nodes
        # Add offset to edge indices so they point to correct nodes in batch
        edge_batch_list.append(graph.edge_index + node_offset)
        # Track which graph each node belongs to
        batch_vector.extend([i] * num_nodes)
        node_offset += num_nodes

    edge_batch = jnp.hstack(edge_batch_list)
    batch_vector = jnp.array(batch_vector)

    # Create batched Data object
    batched_data = Data(x=x_batch, edge_index=edge_batch, y=y_batch, batch=batch_vector)

    return batched_data


def train_step_base(model: SimpleGCN, optimizer: nnx.Optimizer, x, edge_index, y):
    """Base training step with cross-entropy loss."""

    @nnx.value_and_grad
    def grad_loss_fn(model, x, edge_index, y):
        logits = model(x, edge_index)
        # Cross-entropy loss for node classification
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        loss = loss.mean()
        return loss

    # Compute gradients
    loss, grads = grad_loss_fn(model, x, edge_index, y)

    loss, grads = jax.lax.pmean((loss, grads), "dp")

    # Update model parameters
    optimizer.update(model, grads)

    return loss


def create_train_step(mesh):
    """Create sharded training step with given mesh."""

    # Define the sharded version using nnx.shard_map
    train_step_sharded = nnx.shard_map(
        train_step_base,
        mesh=mesh,
        in_specs=(
            P(),  # Model state - replicated across all devices
            P(),  # Optimizer state - replicated across all devices
            P("dp", None),  # x - nodes sharded, features replicated
            P(None, "dp"),  # edge_index - edges sharded
            P("dp"),  # y - labels sharded
        ),
        out_specs=P(),  # Loss - reduced and replicated
    )

    # Apply JIT compilation
    return nnx.jit(train_step_sharded)


def eval_step_base(model: SimpleGCN, x, edge_index, y):
    """Base evaluation step with accuracy computation."""
    model.eval()
    logits = model(x, edge_index)

    # Compute loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    loss = loss.mean()

    # Compute accuracy
    predictions = jnp.argmax(logits, axis=1)
    accuracy = jnp.mean(predictions == y)

    loss, accuracy = jax.lax.pmean((loss, accuracy), "dp")

    return loss, accuracy


def create_eval_step(mesh):
    """Create sharded evaluation step with given mesh."""

    # Define the sharded version using nnx.shard_map
    eval_step_sharded = nnx.shard_map(
        eval_step_base,
        mesh=mesh,
        in_specs=(
            P(),  # Model state - replicated
            P("dp", None),  # x - nodes sharded, features replicated
            P(None, "dp"),  # edge_index - edges sharded
            P("dp"),  # y - labels sharded
        ),
        out_specs=(
            P(),  # Loss - reduced and replicated
            P(),  # Accuracy - reduced and replicated
        ),
    )

    # Apply JIT compilation
    return nnx.jit(eval_step_sharded)


def main():
    """
    Demonstrate training a GCN on synthetic graph data using jraphx.
    """
    print("Graph Convolutional Network - JraphX Example")
    print("=" * 50)

    # Set random seed for reproducibility
    # Create model
    rngs = nnx.Rngs(42)

    model = SimpleGCN(in_features=32, hidden_features=64, out_features=4, rngs=rngs)  # 4 classes

    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    print("\n1. Single Graph Processing with JraphX")
    print("-" * 30)

    # Create a single graph using jraphx Data
    single_graph = create_synthetic_graph_data(rngs, num_nodes=20)
    print(f"Graph: {single_graph}")
    print(f"Number of nodes: {single_graph.num_nodes}")
    print(f"Number of edges: {single_graph.num_edges}")

    # Forward pass
    output = model(single_graph.x, single_graph.edge_index)
    print(f"Output shape: {output.shape} (one prediction per node)")

    print("\n2. Batch Processing with JraphX")
    print("-" * 30)

    # Create multiple graphs
    graphs = []
    for i in range(4):
        graphs.append(create_synthetic_graph_data(rngs, num_nodes=15 + i * 5))

    # Batch them together using jraphx
    batched_data = batch_graphs_jraphx(graphs)
    print(f"Batch contains {len(graphs)} graphs")
    print(f"Batched data: {batched_data}")
    print(f"Total nodes: {batched_data.num_nodes}, Total edges: {batched_data.num_edges}")

    print("\n3. Training with Data Parallelism")
    print("-" * 30)
    print("Using nnx.shard_map for data-parallel training across devices")
    devices = jax.devices()
    print(f"Available devices: {devices}")

    # Create mesh for data parallelism
    mesh = Mesh(devices, axis_names=("dp",))
    print(f"Created mesh with shape: {mesh.shape}")

    # Create sharded training and evaluation functions
    train_step = create_train_step(mesh)
    eval_step = create_eval_step(mesh)

    # Training loop
    model.train()
    num_epochs = 100

    for epoch in range(num_epochs):
        # Generate new random batch each epoch
        graphs = [create_synthetic_graph_data(rngs) for _ in range(4)]
        batched_data = batch_graphs_jraphx(graphs)

        # Train step with jraphx Data
        loss = train_step(model, optimizer, batched_data.x, batched_data.edge_index, batched_data.y)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")

    print("\n4. Evaluation with JraphX")
    print("-" * 30)

    # Test on new graphs
    test_graphs = [create_synthetic_graph_data(rngs, num_nodes=25) for _ in range(4)]
    test_batch = batch_graphs_jraphx(test_graphs)

    # Evaluate
    test_loss, test_accuracy = eval_step(model, test_batch.x, test_batch.edge_index, test_batch.y)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    print("\n5. Using JraphX Features")
    print("-" * 30)

    # Demonstrate accessing graph properties
    example_graph = test_graphs[0]
    print("Graph properties from jraphx Data:")
    print(f"  - num_nodes: {example_graph.num_nodes}")
    print(f"  - num_edges: {example_graph.num_edges}")
    print(f"  - num_node_features: {example_graph.num_node_features}")
    print(f"  - is_directed: {example_graph.is_directed}")

    print("\nExample complete!")
    print("\nKey differences from standalone implementation:")
    print("1. Uses jraphx.GCNConv instead of custom implementation")
    print("2. Uses jraphx.Data for graph representation")
    print("3. Built-in normalization and self-loops in GCNConv")
    print("4. Cleaner API with less boilerplate code")


if __name__ == "__main__":
    main()
