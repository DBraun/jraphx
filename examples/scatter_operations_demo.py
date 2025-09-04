"""Demonstration of scatter operations for GNNs.

This example shows how to use the scatter operations including the newly added:
- scatter_std: Standard deviation aggregation
- scatter_logsumexp: Numerically stable log-sum-exp
- scatter_softmax: Attention weights normalization
- scatter_log_softmax: Log-space softmax for numerical stability
"""

import jax.numpy as jnp
from absl import logging
from jax.scipy.special import logsumexp as jax_logsumexp

from jraphx.utils import (
    scatter_add,
    scatter_log_softmax,
    scatter_logsumexp,
    scatter_max,
    scatter_mean,
    scatter_min,
    scatter_softmax,
    scatter_std,
)


def demo_basic_scatter():
    """Demonstrate basic scatter operations."""
    logging.info("=" * 60)
    logging.info("Basic Scatter Operations")
    logging.info("=" * 60)

    # Example: Aggregating node features by graph membership
    # Imagine we have 5 nodes belonging to 2 graphs
    node_features = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    graph_membership = jnp.array([0, 0, 1, 1, 1])  # First 2 nodes in graph 0, last 3 in graph 1
    num_graphs = 2

    logging.info(f"\nNode features: {node_features}")
    logging.info(f"Graph membership: {graph_membership}")

    # Different aggregations
    sum_result = scatter_add(node_features, graph_membership, dim_size=num_graphs)
    mean_result = scatter_mean(node_features, graph_membership, dim_size=num_graphs)
    max_result = scatter_max(node_features, graph_membership, dim_size=num_graphs)
    min_result = scatter_min(node_features, graph_membership, dim_size=num_graphs)
    std_result = scatter_std(node_features, graph_membership, dim_size=num_graphs)

    logging.info("\nAggregation results per graph:")
    logging.info(f"  Sum:  {sum_result}")  # [3, 12]
    logging.info(f"  Mean: {mean_result}")  # [1.5, 4.0]
    logging.info(f"  Max:  {max_result}")  # [2, 5]
    logging.info(f"  Min:  {min_result}")  # [1, 3]
    logging.info(f"  Std:  {std_result}")  # [0.5, 0.816...]


def demo_attention_softmax():
    """Demonstrate softmax operations for attention mechanisms."""
    logging.info("\n" + "=" * 60)
    logging.info("Attention Softmax Operations")
    logging.info("=" * 60)

    # Example: Computing attention weights in GAT
    # We have attention scores between nodes and their neighbors
    attention_scores = jnp.array([0.5, 1.2, 0.3, 2.0, 1.5, 0.8])
    # Each score belongs to a source node (which node's neighborhood)
    source_nodes = jnp.array([0, 0, 0, 1, 1, 2])
    num_nodes = 3

    logging.info(f"\nAttention scores: {attention_scores}")
    logging.info(f"Source nodes: {source_nodes}")

    # Compute softmax to get attention weights
    attention_weights = scatter_softmax(attention_scores, source_nodes, dim_size=num_nodes)

    logging.info(f"\nAttention weights (softmax): {attention_weights}")
    logging.info(f"  Node 0 weights sum: {attention_weights[:3].sum():.4f}")
    logging.info(f"  Node 1 weights sum: {attention_weights[3:5].sum():.4f}")
    logging.info(f"  Node 2 weights sum: {attention_weights[5:6].sum():.4f}")

    # Log-softmax for numerical stability in loss computation
    log_attention = scatter_log_softmax(attention_scores, source_nodes, dim_size=num_nodes)
    logging.info(f"\nLog attention weights: {log_attention}")

    # Temperature scaling for sharper/smoother attention
    sharp_attention = scatter_softmax(
        attention_scores, source_nodes, dim_size=num_nodes, temperature=0.5
    )
    smooth_attention = scatter_softmax(
        attention_scores, source_nodes, dim_size=num_nodes, temperature=2.0
    )

    logging.info("\nTemperature effects:")
    logging.info(f"  Sharp (T=0.5):  {sharp_attention[:3]}")
    logging.info(f"  Normal (T=1.0): {attention_weights[:3]}")
    logging.info(f"  Smooth (T=2.0): {smooth_attention[:3]}")


def demo_numerical_stability():
    """Demonstrate numerical stability of logsumexp."""
    logging.info("\n" + "=" * 60)
    logging.info("Numerical Stability with LogSumExp")
    logging.info("=" * 60)

    # Large values that would overflow with naive exp->sum->log
    large_values = jnp.array([100.0, 101.0, 102.0, 200.0, 201.0])
    groups = jnp.array([0, 0, 0, 1, 1])

    logging.info(f"\nLarge values: {large_values}")
    logging.info(f"Groups: {groups}")

    # scatter_logsumexp handles this gracefully
    logsumexp_result = scatter_logsumexp(large_values, groups, dim_size=2)
    logging.info(f"\nLogSumExp result: {logsumexp_result}")

    # This is numerically stable and won't overflow
    # Compare with naive approach (would overflow without the max trick)

    group0_vals = large_values[:3]
    group1_vals = large_values[3:]

    expected_0 = jax_logsumexp(group0_vals)
    expected_1 = jax_logsumexp(group1_vals)

    logging.info(f"Expected values: [{expected_0:.4f}, {expected_1:.4f}]")
    logging.info(f"Match: {jnp.allclose(logsumexp_result, jnp.array([expected_0, expected_1]))}")


def demo_graph_pooling():
    """Demonstrate using scatter operations for graph pooling."""
    logging.info("\n" + "=" * 60)
    logging.info("Graph Pooling with Scatter Operations")
    logging.info("=" * 60)

    # Example: Pool node features to graph level
    # 2 graphs with 3 and 2 nodes respectively
    node_features = jnp.array(
        [
            [1.0, 2.0],  # Graph 0, Node 0
            [3.0, 4.0],  # Graph 0, Node 1
            [5.0, 6.0],  # Graph 0, Node 2
            [7.0, 8.0],  # Graph 1, Node 0
            [9.0, 10.0],  # Graph 1, Node 1
        ]
    )

    batch = jnp.array([0, 0, 0, 1, 1])
    num_graphs = 2

    logging.info(f"\nNode features shape: {node_features.shape}")
    logging.info("Node features:")
    for i, feat in enumerate(node_features):
        logging.info(f"  Node {i} (Graph {batch[i]}): {feat}")

    # Different pooling strategies
    mean_pool = scatter_mean(node_features, batch, dim_size=num_graphs)
    max_pool = scatter_max(node_features, batch, dim_size=num_graphs)
    sum_pool = scatter_add(node_features, batch, dim_size=num_graphs)

    logging.info("\nGraph-level representations:")
    logging.info(f"  Mean pooling: {mean_pool}")
    logging.info(f"  Max pooling:  {max_pool}")
    logging.info(f"  Sum pooling:  {sum_pool}")

    # Advanced: Use standard deviation as a measure of feature diversity
    std_pool = scatter_std(node_features, batch, dim_size=num_graphs)
    logging.info(f"  Std pooling:  {std_pool}")
    logging.info("\n  (Std pooling captures feature diversity within each graph)")


def main():
    logging.set_verbosity(logging.INFO)

    logging.info("🚀 JAX Scatter Operations for GNNs")
    logging.info("=" * 60)

    # Run demonstrations
    demo_basic_scatter()
    demo_attention_softmax()
    demo_numerical_stability()
    demo_graph_pooling()

    logging.info("\n" + "=" * 60)
    logging.info("✅ All demonstrations completed successfully!")
    logging.info("\nThese operations are essential building blocks for:")
    logging.info("  - Graph Neural Networks (GCN, GAT, GraphSAGE)")
    logging.info("  - Attention mechanisms")
    logging.info("  - Graph pooling layers")
    logging.info("  - Message passing algorithms")


if __name__ == "__main__":
    main()
