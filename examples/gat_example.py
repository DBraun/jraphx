"""Example demonstrating GAT and GATv2 features in JraphX.

This example shows:
1. Standard GAT vs GATv2 comparison
2. Edge feature handling
3. Bipartite graph support
4. Residual connections
5. Attention weight visualization
6. Performance comparisons
"""

import time

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jraphx import Data  # noqa: E402
from jraphx.nn.conv import GATConv, GATv2Conv  # noqa: E402
from jraphx.nn.norm import LayerNorm  # noqa: E402


def create_karate_club_graph():
    """Create the Zachary's Karate Club graph for demonstration."""
    # Simplified version with 10 nodes for clarity
    edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 4],
        [3, 5],
        [3, 6],
        [4, 5],
        [4, 7],
        [5, 6],
        [5, 7],
        [5, 8],
        [6, 7],
        [6, 8],
        [6, 9],
        [7, 8],
        [7, 9],
        [8, 9],
    ]

    edge_index = jnp.array(edges).T
    # Make bidirectional
    edge_index = jnp.concatenate([edge_index, jnp.array([edge_index[1], edge_index[0]])], axis=1)

    # Node features (random for demonstration)
    x = jnp.array(np.random.randn(10, 16))

    # Edge features (e.g., edge weights/types)
    num_edges = edge_index.shape[1]
    edge_attr = jnp.ones((num_edges, 4))  # 4-dim edge features

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def demo_gat_vs_gatv2():
    """Demonstrate differences between GAT and GATv2."""
    print("=" * 60)
    print("GAT vs GATv2 Comparison")
    print("=" * 60)

    data = create_karate_club_graph()
    rngs = nnx.Rngs(0)

    # Create both models with same parameters
    gat = GATConv(
        in_features=16,
        out_features=8,
        heads=4,
        concat=True,
        edge_dim=4,
        residual=True,
        rngs=rngs,
    )

    rngs = nnx.Rngs(0)  # Reset for same initialization where possible
    gatv2 = GATv2Conv(
        in_features=16,
        out_features=8,
        heads=4,
        concat=True,
        edge_dim=4,
        residual=True,
        share_weights=False,
        rngs=rngs,
    )

    # Forward pass
    out_gat = gat(data.x, data.edge_index, data.edge_attr)
    out_gatv2 = gatv2(data.x, data.edge_index, data.edge_attr)

    print(f"GAT output shape: {out_gat.shape}")
    print(f"GATv2 output shape: {out_gatv2.shape}")
    print(f"Output difference (L2): {jnp.linalg.norm(out_gat - out_gatv2):.4f}")

    # Get attention weights for visualization
    out_gat_with_att, (_, alpha_gat) = gat(
        data.x, data.edge_index, data.edge_attr, return_attention_weights=True
    )
    out_gatv2_with_att, (_, alpha_gatv2) = gatv2(
        data.x, data.edge_index, data.edge_attr, return_attention_weights=True
    )

    print(f"\nGAT attention shape: {alpha_gat.shape}")
    print(f"GATv2 attention shape: {alpha_gatv2.shape}")

    # Compare attention patterns
    print("\nAttention statistics:")
    print(f"GAT   - Mean: {alpha_gat.mean():.4f}, Std: {alpha_gat.std():.4f}")
    print(f"GATv2 - Mean: {alpha_gatv2.mean():.4f}, Std: {alpha_gatv2.std():.4f}")


def demo_bipartite_graph():
    """Demonstrate GAT on bipartite graphs."""
    print("\n" + "=" * 60)
    print("Bipartite Graph Support")
    print("=" * 60)

    # Create a bipartite graph (e.g., users and items)
    num_users = 5
    num_items = 8

    # User features (different dimension)
    x_users = jnp.array(np.random.randn(num_users, 16))
    # Item features
    x_items = jnp.array(np.random.randn(num_items, 32))

    # Edges from users to items (user-item interactions)
    edges = [
        [0, 0],
        [0, 2],
        [0, 5],
        [1, 1],
        [1, 3],
        [1, 7],
        [2, 0],
        [2, 4],
        [2, 6],
        [3, 2],
        [3, 3],
        [3, 5],
        [4, 1],
        [4, 4],
        [4, 6],
        [4, 7],
    ]
    edge_index = jnp.array(edges).T

    print(f"User features shape: {x_users.shape}")
    print(f"Item features shape: {x_items.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")

    # Create GAT for bipartite graph
    rngs = nnx.Rngs(42)
    gat_bipartite = GATConv(
        in_features=(16, 32),  # (source_dim, target_dim)
        out_features=8,
        heads=2,
        concat=True,
        rngs=rngs,
    )

    # Forward pass
    out = gat_bipartite((x_users, x_items), edge_index)

    print(f"\nOutput shape (items): {out.shape}")
    print(f"Expected: ({num_items}, {2 * 8})")


def demo_edge_features():
    """Demonstrate edge feature handling."""
    print("\n" + "=" * 60)
    print("Edge Feature Support")
    print("=" * 60)

    data = create_karate_club_graph()

    # Create edge features with different meanings
    num_edges = data.edge_index.shape[1]

    # Example: [edge_weight, edge_type, time_delta, interaction_count]
    edge_features = jnp.array(
        [
            [np.random.rand(), np.random.randint(0, 3), np.random.rand(), np.random.randint(1, 10)]
            for _ in range(num_edges)
        ]
    )

    print(f"Edge features shape: {edge_features.shape}")

    # GAT without edge features
    rngs = nnx.Rngs(0)
    gat_no_edge = GATConv(
        in_features=16,
        out_features=8,
        heads=3,
        rngs=rngs,
    )

    # GAT with edge features
    rngs = nnx.Rngs(0)
    gat_with_edge = GATConv(
        in_features=16,
        out_features=8,
        heads=3,
        edge_dim=4,
        rngs=rngs,
    )

    # Compare outputs
    out_no_edge = gat_no_edge(data.x, data.edge_index)
    out_with_edge = gat_with_edge(data.x, data.edge_index, edge_features)

    print(f"\nOutput without edge features: {out_no_edge.shape}")
    print(f"Output with edge features: {out_with_edge.shape}")
    print(f"Difference in outputs: {jnp.linalg.norm(out_no_edge - out_with_edge):.4f}")


def demo_residual_connections():
    """Demonstrate residual connections for deeper networks."""
    print("\n" + "=" * 60)
    print("Residual Connections")
    print("=" * 60)

    data = create_karate_club_graph()

    # Deep GAT with optional normalization
    class DeepGAT(nnx.Module):
        def __init__(self, use_residual=False, use_norm=False, rngs=None):
            self.layers = [
                GATConv(16, 16, heads=1, concat=False, residual=use_residual, rngs=rngs),
                GATConv(16, 16, heads=1, concat=False, residual=use_residual, rngs=rngs),
                GATConv(16, 16, heads=1, concat=False, residual=use_residual, rngs=rngs),
            ]
            self.use_norm = use_norm
            if use_norm:
                self.norms = [
                    LayerNorm(16),
                    LayerNorm(16),
                    LayerNorm(16),
                ]

        def __call__(self, x, edge_index):
            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index)
                if self.use_norm:
                    x = self.norms[i](x)
                x = nnx.relu(x)
            return x

    rngs = nnx.Rngs(0)
    model_no_res = DeepGAT(use_residual=False, use_norm=False, rngs=rngs)

    rngs = nnx.Rngs(0)
    model_with_res = DeepGAT(use_residual=True, use_norm=False, rngs=rngs)

    rngs = nnx.Rngs(0)
    model_with_norm = DeepGAT(use_residual=True, use_norm=True, rngs=rngs)

    # Forward pass
    out_no_res = model_no_res(data.x, data.edge_index)
    out_with_res = model_with_res(data.x, data.edge_index)
    out_with_norm = model_with_norm(data.x, data.edge_index)

    print(f"Output norm without residual: {jnp.linalg.norm(out_no_res):.4f}")
    print(f"Output norm with residual: {jnp.linalg.norm(out_with_res):.4f}")
    print(f"Output norm with residual + LayerNorm: {jnp.linalg.norm(out_with_norm):.4f}")

    # Check gradient flow
    def loss_fn(model, x, edge_index):
        out = model(x, edge_index)
        return jnp.mean(out**2)

    # Compute gradients
    graphdef_no_res, state_no_res = nnx.split(model_no_res)
    graphdef_with_res, state_with_res = nnx.split(model_with_res)
    graphdef_with_norm, state_with_norm = nnx.split(model_with_norm)

    def compute_grad_norm(graphdef, state):
        def stateful_loss(state):
            model = nnx.merge(graphdef, state)
            return loss_fn(model, data.x, data.edge_index)

        _, grads = jax.value_and_grad(stateful_loss)(state)
        grad_norm = jax.tree.map(lambda g: jnp.linalg.norm(g), grads)
        return jax.tree.reduce(lambda x, y: x + y, grad_norm)

    grad_norm_no_res = compute_grad_norm(graphdef_no_res, state_no_res)
    grad_norm_with_res = compute_grad_norm(graphdef_with_res, state_with_res)
    grad_norm_with_norm = compute_grad_norm(graphdef_with_norm, state_with_norm)

    print(f"\nGradient norm without residual: {grad_norm_no_res:.6f}")
    print(f"Gradient norm with residual: {grad_norm_with_res:.6f}")
    print(f"Gradient norm with residual + LayerNorm: {grad_norm_with_norm:.6f}")


def demo_performance():
    """Benchmark performance of different configurations."""
    print("\n" + "=" * 60)
    print("Performance Benchmarks")
    print("=" * 60)

    # Create larger graph for benchmarking
    num_nodes = 1000
    num_edges = 5000

    edge_index = jnp.array(np.random.randint(0, num_nodes, (2, num_edges)))
    x = jnp.array(np.random.randn(num_nodes, 64))
    edge_attr = jnp.array(np.random.randn(num_edges, 8))

    rngs = nnx.Rngs(0)

    configs = [
        ("GAT (4 heads)", GATConv(64, 16, heads=4, rngs=rngs), False),
        ("GAT (8 heads)", GATConv(64, 16, heads=8, rngs=rngs), False),
        ("GAT + Edge", GATConv(64, 16, heads=4, edge_dim=8, rngs=rngs), True),
        ("GATv2 (4 heads)", GATv2Conv(64, 16, heads=4, rngs=rngs), False),
        ("GATv2 + Shared", GATv2Conv(64, 16, heads=4, share_weights=True, rngs=rngs), False),
    ]

    print(f"Graph size: {num_nodes} nodes, {num_edges} edges")
    print(f"Feature dim: {x.shape[1]}")
    print("\nBenchmark results (forward pass):")

    # Define the jitted forward function outside the loop
    @jax.jit
    def run_forward_no_edge(state_and_graphdef, x, edge_index):
        graphdef, state = state_and_graphdef
        model = nnx.merge(graphdef, state)
        return model(x, edge_index)

    @jax.jit
    def run_forward_with_edge(state_and_graphdef, x, edge_index, edge_attr):
        graphdef, state = state_and_graphdef
        model = nnx.merge(graphdef, state)
        return model(x, edge_index, edge_attr)

    for name, model, use_edge in configs:
        graphdef, state = nnx.split(model)
        state_and_graphdef = (graphdef, state)

        # Warm-up
        if use_edge:
            _ = run_forward_with_edge(state_and_graphdef, x, edge_index, edge_attr)
            _ = run_forward_with_edge(state_and_graphdef, x, edge_index, edge_attr)
        else:
            _ = run_forward_no_edge(state_and_graphdef, x, edge_index)
            _ = run_forward_no_edge(state_and_graphdef, x, edge_index)

        # Benchmark
        start = time.time()
        for _ in range(10):
            if use_edge:
                out = run_forward_with_edge(state_and_graphdef, x, edge_index, edge_attr)
            else:
                out = run_forward_no_edge(state_and_graphdef, x, edge_index)
            out.block_until_ready()
        elapsed = (time.time() - start) / 10

        print(f"{name:20} - {elapsed*1000:.2f} ms/forward")


def visualize_attention():
    """Visualize attention weights for a small graph."""
    print("\n" + "=" * 60)
    print("Attention Weight Visualization")
    print("=" * 60)

    # Create small graph for visualization
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]
    edge_index = jnp.array(edges).T
    edge_index = jnp.concatenate([edge_index, jnp.array([edge_index[1], edge_index[0]])], axis=1)

    x = jnp.array(np.random.randn(4, 8))

    # Create models
    rngs = nnx.Rngs(0)
    gat = GATConv(8, 4, heads=2, rngs=rngs)

    rngs = nnx.Rngs(0)
    gatv2 = GATv2Conv(8, 4, heads=2, rngs=rngs)

    # Get attention weights
    _, (edges_gat, alpha_gat) = gat(x, edge_index, return_attention_weights=True)
    _, (edges_gatv2, alpha_gatv2) = gatv2(x, edge_index, return_attention_weights=True)

    print(f"Number of edges (with self-loops): {edges_gat.shape[1]}")
    print(f"Attention weights shape: {alpha_gat.shape}")

    # Print attention matrix for first head
    print("\nGAT Attention (Head 0):")
    for i in range(edges_gat.shape[1]):
        src, dst = edges_gat[0, i], edges_gat[1, i]
        att = alpha_gat[i, 0]
        print(f"  Edge {src} -> {dst}: {att:.4f}")

    print("\nGATv2 Attention (Head 0):")
    for i in range(edges_gatv2.shape[1]):
        src, dst = edges_gatv2[0, i], edges_gatv2[1, i]
        att = alpha_gatv2[i, 0]
        print(f"  Edge {src} -> {dst}: {att:.4f}")


def main():
    """Run all demonstrations."""
    print("🚀 Comprehensive GAT and GATv2 Demonstration")
    print("=" * 60)

    # Run all demos
    demo_gat_vs_gatv2()
    demo_bipartite_graph()
    demo_edge_features()
    demo_residual_connections()
    visualize_attention()
    demo_performance()

    print("\n" + "=" * 60)
    print("✅ All demonstrations completed successfully!")
    print("\nKey takeaways:")
    print("1. GATv2 fixes the 'static attention' problem of GAT")
    print("2. Edge features can significantly impact attention patterns")
    print("3. Residual connections help with gradient flow in deep networks")
    print("4. Bipartite graphs are naturally supported")
    print("5. JIT compilation provides significant speedups")


if __name__ == "__main__":
    main()
