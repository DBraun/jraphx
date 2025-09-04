"""Examples demonstrating NNX transformations with JraphX.

This module shows how to use Flax NNX's transformations (vmap, scan, jit, grad)
for efficient graph neural network processing, including parallel batch processing,
sequential layer scanning, and memory-efficient training.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx.nn.recurrent import GRUCell

# JraphX imports
from jraphx.nn.conv import GATConv, GCNConv
from jraphx.nn.pool import global_mean_pool


def example_1_vmap_batch_processing():
    """Example 1: Process multiple graphs in parallel with vmap."""
    print("Example 1: Batch processing with vmap")
    print("-" * 40)

    class GNN(nnx.Module):
        def __init__(self, in_features: int, hidden_dim: int, out_dim: int):
            self.conv1 = GCNConv(in_features, hidden_dim, rngs=nnx.Rngs(0))
            self.conv2 = GCNConv(hidden_dim, out_dim, rngs=nnx.Rngs(1))

        def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
            x = self.conv1(x, edge_index)
            x = nnx.relu(x)
            x = self.conv2(x, edge_index)
            return global_mean_pool(x, batch=None)

    # Create model and batch data
    model = GNN(16, 32, 8)
    batch_size = 4
    num_nodes, num_edges = 10, 20

    rngs = nnx.Rngs(42)
    x_batch = rngs.normal((batch_size, num_nodes, 16))
    edge_batch = rngs.choice(num_nodes, (batch_size, 2, num_edges))

    # Show model architecture
    print("\nModel architecture:")
    print(nnx.tabulate(model, x_batch[0], edge_batch[0], depth=1))

    # Process all graphs in parallel
    outputs = jax.vmap(model)(x_batch, edge_batch)

    print(f"\nInput shape: {x_batch.shape}")  # (4, 10, 16)
    print(f"Output shape: {outputs.shape}")  # (4, 1, 8)
    print()


def example_2_scan_sequential_layers():
    """Example 2: Use nnx.scan for sequential GNN layers."""
    print("Example 2: Sequential layers with nnx.scan")
    print("-" * 40)

    class StackedGNN(nnx.Module):
        def __init__(self, in_features: int, hidden_dim: int, num_layers: int):
            # Create initial layer separately (different input dim)
            self.first_layer = GCNConv(in_features, hidden_dim, rngs=nnx.Rngs(0))

            # Create stacked layers using scan
            @nnx.split_rngs(splits=num_layers - 1)
            @nnx.scan(in_axes=0, out_axes=0, length=num_layers - 1)
            def create_layers(rngs: nnx.Rngs):
                return GCNConv(hidden_dim, hidden_dim, rngs=rngs)

            self.layers = create_layers(nnx.Rngs(1))

        def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
            # Apply first layer
            x = self.first_layer(x, edge_index)
            x = nnx.relu(x)

            # Apply remaining layers using scan
            @nnx.scan(in_axes=(0, nnx.Carry, None), out_axes=nnx.Carry)
            def apply_layer(layer: GCNConv, x: jnp.ndarray, edge_index: jnp.ndarray):
                x = layer(x, edge_index)
                x = nnx.relu(x)
                return x

            x = apply_layer(self.layers, x, edge_index)
            return x

    # Create model and test
    model = StackedGNN(16, 32, num_layers=5)
    rngs = nnx.Rngs(42)
    x = rngs.normal((20, 16))
    edge_index = rngs.choice(20, (2, 50))

    # Show model architecture with stacked layers
    print("\nStacked GNN architecture:")
    print(nnx.tabulate(model, x, edge_index, depth=1))

    output = model(x, edge_index)
    print(f"\nInput shape: {x.shape}")  # (20, 16)
    print(f"Output shape: {output.shape}")  # (20, 32)
    print()


def example_3_temporal_graph_networks():
    """Example 3: Temporal Graph Networks using vmap + scan.

    This example shows two approaches:
    1. Custom GraphGRUCell following NNX RNNCellBase pattern
    2. Using NNX's built-in GRUCell with graph operations
    """
    print("Example 3: Temporal GNN with vmap + scan")
    print("-" * 40)

    class GraphGRUCell(nnx.Module):
        """Graph-based GRU cell following NNX RNNCellBase pattern.

        Simplified GRU formulation for graphs:
        .. math::
            m_t = GraphConv(x_t, edge_index)  # Graph message passing
            y_t = [h_{t-1}, m_t]  # Concatenate hidden and message
            z_t = σ(W_z · y_t + b_z)  # Update gate
            r_t = σ(W_r · y_t + b_r)  # Reset gate
            ñ_t = tanh(W_n · [r_t ⊙ h_{t-1}, m_t] + b_n)  # Candidate state
            h_t = (1 - z_t) ⊙ ñ_t + z_t ⊙ h_{t-1}  # Final hidden state
        """

        def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
            self.in_features = in_features
            self.hidden_features = hidden_features

            # Graph convolution for message passing
            self.conv = GCNConv(in_features, hidden_features, rngs=rngs)

            # Combined linear layer for all GRU gates (more efficient)
            # Output: [reset_gate, update_gate, candidate]
            self.gates = nnx.Linear(
                hidden_features * 2,  # hidden + message
                hidden_features * 3,  # r, z, n gates
                rngs=rngs,
            )

        def __call__(self, carry: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
            """Process one timestep of the Graph GRU.

            Args:
                carry: Hidden state from previous timestep [num_nodes, hidden_features]
                inputs: Tuple of (node_features, edge_index)

            Returns:
                Tuple of (new_carry, output)
            """
            x_t, edge_index = inputs
            h_prev = carry

            # Graph message passing
            msg = self.conv(x_t, edge_index)

            # Combine hidden state and message
            combined = jnp.concatenate([h_prev, msg], axis=-1)

            # Compute reset and update gates
            gates_output = self.gates(combined)
            r_gate, z_gate, _ = jnp.split(gates_output, 3, axis=-1)

            # Apply activations to gates
            r = nnx.sigmoid(r_gate)  # Reset gate
            z = nnx.sigmoid(z_gate)  # Update gate

            # Apply reset gate and compute candidate
            h_reset = r * h_prev
            combined_reset = jnp.concatenate([h_reset, msg], axis=-1)
            # We need a separate transformation for the candidate
            # For simplicity, reuse gates layer's last third
            n = nnx.tanh(self.gates(combined_reset)[:, -self.hidden_features :])

            # Final GRU update
            h_new = (1 - z) * n + z * h_prev

            return h_new, h_new

        def initialize_carry(self, batch_shape: tuple[int, ...]) -> jnp.ndarray:
            """Initialize the hidden state.

            Args:
                batch_shape: Shape of the batch (e.g., (num_nodes,))

            Returns:
                Initialized hidden state
            """
            return jnp.zeros(batch_shape + (self.hidden_features,))

    # Setup data
    batch_size = 4
    num_timesteps = 10
    num_nodes = 15
    feature_dim = 16

    rng = nnx.Rngs(42)
    # Shape: (time, batch, nodes, features) - time first for scan
    temporal_features = rng.normal((num_timesteps, batch_size, num_nodes, feature_dim))
    edge_index = rng.choice(num_nodes, (2, 30))

    # Approach 1: Custom GraphGRUCell
    print("\nApproach 1: Custom GraphGRUCell")
    graph_cell = GraphGRUCell(feature_dim, feature_dim, rngs=nnx.Rngs(0))

    # Show model architecture
    print("GraphGRUCell architecture:")
    h0_sample = graph_cell.initialize_carry((num_nodes,))
    inputs_sample = (temporal_features[0, 0], edge_index)
    print(nnx.tabulate(graph_cell, h0_sample, inputs_sample, depth=1))

    # Process temporal sequences with scan over time and vmap over batch
    h0_batch = jnp.zeros((batch_size, num_nodes, feature_dim))

    @nnx.scan(in_axes=(None, nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
    def process_graph_timestep(cell, h_batch, x_batch, edges):
        # Process all graphs in batch at this timestep
        def single_graph_step(h, x):
            return cell(h, (x, edges))[0]  # Return only new carry

        h_new = jax.vmap(single_graph_step)(h_batch, x_batch)
        return h_new, h_new

    final_h1, all_h1 = process_graph_timestep(graph_cell, h0_batch, temporal_features, edge_index)
    graph_reprs1 = jax.vmap(lambda h: global_mean_pool(h, batch=None))(final_h1)

    print(f"Output shape: {graph_reprs1.shape}")  # (4, 1, 16)

    # Approach 2: Using NNX's built-in GRUCell with graph preprocessing
    print("\nApproach 2: NNX GRUCell with graph preprocessing")

    # GRUCell imported at top of file

    class GraphPreprocessor(nnx.Module):
        """Preprocesses graph data before feeding to standard RNN cells."""

        def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
            self.conv = GCNConv(in_features, out_features, rngs=rngs)

        def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
            return self.conv(x, edge_index)

    # Create graph preprocessor and standard GRU cell
    preprocessor = GraphPreprocessor(feature_dim, feature_dim, rngs=nnx.Rngs(1))
    gru_cell = GRUCell(feature_dim, feature_dim, rngs=nnx.Rngs(2))

    @nnx.scan(in_axes=(None, None, nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
    def process_with_gru(preprocessor, gru_cell, h_batch, x_batch, edges):
        # Preprocess with graph convolution
        x_graph = jax.vmap(preprocessor, in_axes=(0, None))(x_batch, edges)
        # Apply standard GRU cell
        h_new, y = jax.vmap(gru_cell)(h_batch, x_graph)
        return h_new, y

    h0_gru = gru_cell.initialize_carry((batch_size, num_nodes, feature_dim), nnx.Rngs(3))
    final_h2, all_h2 = process_with_gru(
        preprocessor, gru_cell, h0_gru, temporal_features, edge_index
    )
    graph_reprs2 = jax.vmap(lambda h: global_mean_pool(h, batch=None))(final_h2)

    print(f"Output shape: {graph_reprs2.shape}")  # (4, 1, 16)
    print(f"\nInput shape: {temporal_features.shape}")  # (10, 4, 15, 16)
    print("Both approaches produce temporal graph embeddings!")
    print()


def example_4_memory_efficient_training():
    """Example 4: Memory-efficient training with scan over mini-batches."""
    print("Example 4: Memory-efficient training with scan")
    print("-" * 40)

    class TrainableGNN(nnx.Module):
        def __init__(self, in_dim: int, out_dim: int):
            self.conv1 = GATConv(in_dim, 32, heads=4, rngs=nnx.Rngs(0))
            self.conv2 = GATConv(32 * 4, out_dim, heads=1, rngs=nnx.Rngs(1))

        def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray):
            x = self.conv1(x, edge_index)
            x = nnx.elu(x)
            x = self.conv2(x, edge_index)
            return global_mean_pool(x, batch=None)

    # Prepare mini-batched data
    num_minibatches = 8
    graphs_per_batch = 4
    rng = nnx.Rngs(42)

    # Create batches of graph data
    all_x = rng.normal((num_minibatches, graphs_per_batch, 20, 16))
    all_edges = rng.choice(20, (num_minibatches, graphs_per_batch, 2, 40))
    all_y = rng.normal((num_minibatches, graphs_per_batch, 1, 8))

    # Create model and optimizer
    model = TrainableGNN(16, 8)
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    # Show model architecture
    print("\nTrainable GNN architecture:")
    sample_x = all_x[0, 0]  # Single graph for shape inference
    sample_edge = all_edges[0, 0]
    print(nnx.tabulate(model, sample_x, sample_edge, depth=1))

    @nnx.jit
    def train_epoch(model, optimizer, all_x, all_edges, all_y):
        """Train over all mini-batches using scan for memory efficiency."""

        def loss_fn(model, x_batch, edge_batch, y_batch):
            # Process graphs in parallel within batch
            preds = jax.vmap(model)(x_batch, edge_batch)
            return jnp.mean((preds - y_batch) ** 2)

        # Scan over mini-batches to accumulate loss and update model
        @nnx.scan(in_axes=(None, None, nnx.Carry, 0, 0, 0), out_axes=nnx.Carry)
        def train_step(model, optimizer, total_loss, x_batch, edge_batch, y_batch):
            batch_loss, grads = nnx.value_and_grad(loss_fn)(model, x_batch, edge_batch, y_batch)
            optimizer.update(model, grads)
            return total_loss + batch_loss

        total_loss = train_step(model, optimizer, 0.0, all_x, all_edges, all_y)
        return total_loss / num_minibatches

    # Train for one epoch
    avg_loss = train_epoch(model, optimizer, all_x, all_edges, all_y)

    print(f"\nAverage loss: {avg_loss:.4f}")
    print(f"Processed {num_minibatches} mini-batches with {graphs_per_batch} graphs each")
    print()


def run_all_examples():
    """Run all NNX transformation examples."""
    print("=" * 50)
    print("JraphX NNX Transformations Examples")
    print("=" * 50)
    print()

    example_1_vmap_batch_processing()
    example_2_scan_sequential_layers()
    example_3_temporal_graph_networks()
    example_4_memory_efficient_training()

    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    # Set to use CPU
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"

    run_all_examples()
