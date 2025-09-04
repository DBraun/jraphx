"""
Standalone Graph Convolutional Network (GCN) Example using JAX and Flax NNX
============================================================================

This example demonstrates core Graph Neural Network concepts without importing from jraphx.
It shows how to:
1. Represent graphs using edge indices and node features
2. Implement message passing and aggregation
3. Handle batching of multiple graphs
4. Train a GCN model on graph data
5. Use data parallelism with nnx.shard_map for multi-device training

Key concepts illustrated:
- Message Passing: How information flows between connected nodes
- Aggregation: How messages from neighbors are combined
- Graph Batching: Processing multiple graphs in parallel
- Scatter Operations: Efficient aggregation using JAX primitives
- Data Parallelism: Using nnx.shard_map to distribute training across devices
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


class GCNConv(nnx.Module):
    """
    Graph Convolutional Layer - the core building block of GCNs.

    This layer performs the following operations:
    1. Applies a linear transformation to node features
    2. Propagates transformed features to neighbors
    3. Aggregates neighbor features using normalized summation
    4. Adds bias term

    The normalization uses symmetric normalization: D^(-1/2) * A * D^(-1/2)
    where D is the degree matrix and A is the adjacency matrix with self-loops.
    """

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)
        self.bias = nnx.Param(jnp.zeros((out_features,)))

    def __call__(
        self, x: jnp.ndarray, edge_index: jnp.ndarray, num_nodes: int = None
    ) -> jnp.ndarray:
        """
        Forward pass of GCN layer.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Total number of nodes (optional, inferred from x if not provided)

        Returns:
            Updated node features [num_nodes, out_features]
        """
        if num_nodes is None:
            num_nodes = x.shape[0]

        # Step 1: Add self-loops to ensure nodes aggregate their own features
        self_loops = jnp.arange(num_nodes)
        self_loop_edges = jnp.stack([self_loops, self_loops], axis=0)
        edge_index_with_loops = jnp.hstack([edge_index, self_loop_edges])

        # Step 2: Compute node degrees for normalization
        # Degree = number of edges connected to each node
        row, col = edge_index_with_loops[0], edge_index_with_loops[1]
        deg = jax.ops.segment_sum(
            jnp.ones_like(row, dtype=jnp.float32), row, num_segments=num_nodes
        )

        # Compute normalization factors: 1/sqrt(degree)
        deg_inv_sqrt = jnp.power(deg, -0.5)
        deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0.0, deg_inv_sqrt)

        # Step 3: Normalize edge weights using symmetric normalization
        # weight_ij = 1 / (sqrt(deg_i) * sqrt(deg_j))
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Apply linear transformation to features
        x = self.linear(x)

        # Step 5: Message passing - multiply features by edge weights and aggregate
        # For each edge (i->j), send message x_i * weight_ij to node j
        messages = x[row] * edge_weight[:, None]

        # Aggregate messages at target nodes using summation
        out = jax.ops.segment_sum(messages, col, num_segments=num_nodes)

        return out + self.bias


class SimpleGCN(nnx.Module):
    """
    A simple 2-layer GCN for node classification.

    Architecture:
        Input -> GCN(hidden_dim) -> ReLU -> Dropout -> GCN(out_dim) -> Output
    """

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, *, rngs: nnx.Rngs
    ):
        self.conv1 = GCNConv(in_features, hidden_features, rngs=rngs)
        self.conv2 = GCNConv(hidden_features, out_features, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = nnx.relu(x)
        x = self.dropout(x)

        # Second GCN layer (no activation for final layer)
        x = self.conv2(x, edge_index)
        return x


def create_synthetic_graph(rngs: nnx.Rngs, num_nodes: int = 20, num_edges: int = 40):
    """
    Create a synthetic graph for demonstration.

    Returns:
        - x: Random node features
        - edge_index: Random edges (connectivity)
        - y: Random node labels (for classification)
    """
    # Random node features (32-dimensional)
    x = rngs.normal((num_nodes, 32))

    # Random edges - sample pairs of nodes
    edges = rngs.choice(num_nodes, shape=(2, num_edges), replace=True)

    # Random labels (4 classes)
    y = rngs.randint((num_nodes,), 0, 4)

    return x, edges, y


def batch_graphs(graphs_list):
    """
    Batch multiple graphs into a single disconnected graph.

    This is a key operation in GNNs - we combine multiple graphs into one
    large graph where each component is independent. This allows processing
    multiple graphs in parallel.

    Args:
        graphs_list: List of (x, edge_index, y) tuples

    Returns:
        Batched graph data and batch assignment vector
    """
    x_list, edge_list, y_list = zip(*graphs_list, strict=False)

    # Stack all node features
    x_batch = jnp.vstack(x_list)
    y_batch = jnp.hstack(y_list)

    # Offset edge indices for each graph
    edge_batch_list = []
    node_offset = 0
    batch_vector = []

    for i, (x, edges) in enumerate(zip(x_list, edge_list, strict=False)):
        num_nodes = x.shape[0]
        # Add offset to edge indices so they point to correct nodes in batch
        edge_batch_list.append(edges + node_offset)
        # Track which graph each node belongs to
        batch_vector.extend([i] * num_nodes)
        node_offset += num_nodes

    edge_batch = jnp.hstack(edge_batch_list)
    batch_vector = jnp.array(batch_vector)

    return x_batch, edge_batch, y_batch, batch_vector


def compute_graph_level_output(node_features, batch_vector, num_graphs):
    """
    Aggregate node features to graph-level representations.

    This demonstrates global pooling - combining all nodes in a graph
    into a single feature vector.

    Args:
        node_features: Features for all nodes in batch
        batch_vector: Assignment of nodes to graphs
        num_graphs: Number of graphs in batch

    Returns:
        Graph-level feature vectors [num_graphs, feature_dim]
    """
    # Mean pooling - average features of all nodes in each graph
    graph_features = jax.ops.segment_sum(node_features, batch_vector, num_segments=num_graphs)

    # Compute number of nodes per graph for averaging
    nodes_per_graph = jax.ops.segment_sum(
        jnp.ones((node_features.shape[0], 1)), batch_vector, num_segments=num_graphs
    )

    return graph_features / nodes_per_graph


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


# Create a function that returns the sharded training step
def create_train_step(mesh):
    """Create sharded training step with given mesh."""

    # Define the sharded version using nnx.shard_map as a function transformation
    train_step_sharded = nnx.shard_map(
        train_step_base,
        mesh=mesh,
        in_specs=(
            P(),  # Model state - replicated across all devices
            P(),  # Optimizer state - replicated across all devices
            P("dp", None),  # x - nodes sharded across devices, features replicated
            P(None, "dp"),  # edge_index - edges sharded across devices
            P("dp"),  # y - labels sharded across devices
        ),
        out_specs=P(),  # Loss - reduced and replicated (scalar)
    )

    # Apply JIT compilation to the sharded function for optimal performance
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


# Create a function that returns the sharded evaluation step
def create_eval_step(mesh):
    """Create sharded evaluation step with given mesh."""

    # Define the sharded version using nnx.shard_map as a function transformation
    eval_step_sharded = nnx.shard_map(
        eval_step_base,
        mesh=mesh,
        in_specs=(
            P(),  # Model state - replicated across all devices
            P("dp", None),  # x - nodes sharded, features replicated
            P(None, "dp"),  # edge_index - edges sharded
            P("dp"),  # y - labels sharded
        ),
        out_specs=(
            P(),  # Loss - reduced and replicated
            P(),  # Accuracy - reduced and replicated
        ),
    )

    # Apply JIT compilation for optimal performance
    return nnx.jit(eval_step_sharded)


def main():
    """
    Demonstrate training a GCN on synthetic graph data.
    """
    print("Graph Convolutional Network - Standalone Example")
    print("=" * 50)

    # Set random seed for reproducibility
    rngs = nnx.Rngs(42)

    # Create model
    model = SimpleGCN(in_features=32, hidden_features=64, out_features=4, rngs=rngs)  # 4 classes

    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    print("\n1. Single Graph Processing")
    print("-" * 30)

    # Create a single graph
    x, edge_index, y = create_synthetic_graph(rngs, num_nodes=20)
    print(f"Graph with {x.shape[0]} nodes and {edge_index.shape[1]} edges")

    # Forward pass
    output = model(x, edge_index)
    print(f"Output shape: {output.shape} (one prediction per node)")

    print("\n2. Batch Processing")
    print("-" * 30)

    # Create multiple graphs
    graphs = []
    for i in range(4):
        graphs.append(create_synthetic_graph(rngs, num_nodes=15 + i * 5))

    # Batch them together
    x_batch, edge_batch, y_batch, batch_vector = batch_graphs(graphs)
    print(f"Batch contains {len(graphs)} graphs")
    print(f"Total nodes: {x_batch.shape[0]}, Total edges: {edge_batch.shape[1]}")

    print("\n3. Training with Data Parallelism")
    print("-" * 30)
    print("Using nnx.shard_map for data-parallel training across devices")
    devices = jax.devices()
    print(f"Available devices: {devices}")

    # Create mesh for data parallelism
    # Arrange devices along a single "dp" (data parallel) axis
    # With multiple GPUs, each device would process a shard of the batch
    # For example, with 4 GPUs and batch_size=16, each GPU processes 4 samples
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
        graphs = [create_synthetic_graph(rngs) for _ in range(4)]
        x_batch, edge_batch, y_batch, _ = batch_graphs(graphs)

        # Note: train_step uses nnx.shard_map internally for data parallelism
        # The batch is automatically distributed across available devices
        loss = train_step(model, optimizer, x_batch, edge_batch, y_batch)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")

    print("\n4. Evaluation with Data Parallelism")
    print("-" * 30)

    # Test on new graphs
    test_graphs = [create_synthetic_graph(rngs, num_nodes=25) for _ in range(4)]
    x_test_batch, edge_test_batch, y_test_batch, _ = batch_graphs(test_graphs)

    # Use eval_step with sharding for evaluation
    test_loss, test_accuracy = eval_step(model, x_test_batch, edge_test_batch, y_test_batch)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    # Demonstrate graph-level pooling
    print("\n5. Graph-Level Representations")
    print("-" * 30)

    # Use a single graph for demonstration
    x_single, edge_single, _ = test_graphs[0]

    # Get node embeddings from first layer
    node_embeddings = model.conv1(x_single, edge_single)
    node_embeddings = nnx.relu(node_embeddings)

    # Create a batch vector for single graph (all zeros)
    batch_vec = jnp.zeros(x_single.shape[0], dtype=jnp.int32)

    # Compute graph-level representation
    graph_repr = compute_graph_level_output(node_embeddings, batch_vec, 1)
    print(f"Graph representation shape: {graph_repr.shape}")
    print("This can be used for graph-level tasks like graph classification")

    print("\nExample complete!")


if __name__ == "__main__":
    main()
