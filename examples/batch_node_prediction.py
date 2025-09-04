"""
Batch Processing with Variable Edge Structures - Node Feature Prediction
========================================================================

This example demonstrates:
1. Creating batches of graphs with fixed node count (6) but variable edges
2. Different edge patterns: cycle, star, complete, random
3. Node feature prediction (regression) using GCN
4. JIT compilation for efficient batch processing
5. Edge padding to ensure consistent shapes for JIT optimization
6. Efficient data loading with grain.DataLoader and multi-worker support

Each graph has exactly 6 nodes but different directed edge connectivity patterns.
To avoid JIT recompilation, all edge indices are padded to a fixed maximum size,
ensuring consistent tensor shapes across all batches for optimal performance.

The example uses Google's grain library for high-performance data loading with:
- RandomAccessDataSource for graph generation using pure NumPy
- Custom BatchGraphs transform for combining graphs into batches
- Multi-worker parallel data loading for CPU-GPU pipeline
"""

import time
from typing import SupportsIndex

import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax import nnx

from jraphx.nn.conv import GCNConv


class BatchGraphs(grain.transforms.Map):
    """Custom batch operation for graph data."""

    def map(self, batch_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Batch multiple graphs into a single disconnected graph.

        Args:
            batch_dict: Dictionary with batched arrays from grain.Batch
                       x: (batch_size, num_nodes, feature_dim)
                       edge_index: (batch_size, 2, max_edges)
                       y: (batch_size, num_nodes, output_dim)

        Returns:
            Batched graph dictionary with combined arrays
        """
        # grain.Batch creates arrays with batch dimension first
        # We need to combine them into a single graph

        # Get shapes
        x_batch = batch_dict["x"]  # (batch_size, num_nodes, feature_dim)
        edge_batch = batch_dict["edge_index"]  # (batch_size, 2, max_edges)
        y_batch = batch_dict["y"]  # (batch_size, num_nodes, output_dim)

        batch_size = x_batch.shape[0]
        num_nodes = x_batch.shape[1]
        feature_dim = x_batch.shape[2]
        max_edges = edge_batch.shape[-1]  # Last dimension is max_edges
        output_dim = y_batch.shape[2]

        # Pre-allocate combined arrays
        batch_x = np.zeros((batch_size * num_nodes, feature_dim), dtype=np.float32)
        batch_edges = np.zeros((2, batch_size * max_edges), dtype=np.int32)
        batch_y = np.zeros((batch_size * num_nodes, output_dim), dtype=np.float32)

        # Combine graphs
        for i in range(batch_size):
            node_offset = i * num_nodes
            edge_offset = i * max_edges

            # Copy node features
            batch_x[node_offset : node_offset + num_nodes] = x_batch[i]
            batch_y[node_offset : node_offset + num_nodes] = y_batch[i]

            # Copy edges with node offset
            # edge_batch[i] is (2, max_edges), need to add node offset
            batch_edges[:, edge_offset : edge_offset + max_edges] = (
                edge_batch[i, :, :] + node_offset
            )

        return {"x": batch_x, "edge_index": batch_edges, "y": batch_y}


class GraphBatchDataSource(grain.sources.RandomAccessDataSource):
    """RandomAccessDataSource for generating batched graphs with pure NumPy.

    All operations use pure NumPy (not JAX) for CPU-based data generation.
    This allows parallel processing with multiple workers.
    """

    def __init__(
        self,
        num_samples: int,
        num_nodes: int = 6,
        feature_dim: int = 16,
        output_dim: int = 8,
        seed: int = 42,
    ):
        """Initialize the data source.

        Args:
            num_samples: Total number of samples in the dataset
            num_nodes: Number of nodes per graph
            feature_dim: Input feature dimension
            output_dim: Output feature dimension
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.seed = seed

        # Max edges for padding (complete graph has the most)
        self.max_edges_per_graph = num_nodes * (num_nodes - 1)  # 30 for 6 nodes
        self.total_nodes = num_nodes  # 6 nodes per graph
        self.total_edges = self.max_edges_per_graph  # 30 edges max per graph

        self.graph_types = ["cycle", "star", "complete", "random"]

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: SupportsIndex) -> dict[str, np.ndarray]:
        """Generate a single graph using pure NumPy.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys 'x', 'edge_index', 'y' containing NumPy arrays
        """
        idx = int(idx)

        # Create deterministic random state based on index
        rng = np.random.RandomState(self.seed + idx)

        # Choose graph type cyclically based on index
        graph_type = self.graph_types[idx % len(self.graph_types)]

        # Generate edges based on type
        if graph_type == "cycle":
            edges = self._create_cycle_edges()
        elif graph_type == "star":
            edges = self._create_star_edges()
        elif graph_type == "complete":
            edges = self._create_complete_edges()
        else:  # random
            edges = self._create_random_edges(rng)

        num_edges = edges.shape[1]

        # Pre-allocate edge array with padding
        edge_index = np.zeros((2, self.total_edges), dtype=np.int32)
        edge_index[:, :num_edges] = edges

        # Pad with self-loops on node 0
        if num_edges < self.max_edges_per_graph:
            edge_index[:, num_edges:] = 0

        # Generate node features
        x = rng.randn(self.num_nodes, self.feature_dim).astype(np.float32)

        # Generate target features (nonlinear transformation for regression task)
        W = rng.randn(self.feature_dim, self.output_dim).astype(np.float32) * 0.1
        noise = rng.randn(self.num_nodes, self.output_dim).astype(np.float32) * 0.1
        y = np.tanh(x @ W) + noise

        return {"x": x, "edge_index": edge_index, "y": y}

    def _create_cycle_edges(self) -> np.ndarray:
        """Create directed cycle graph edges."""
        edges = np.zeros((2, self.num_nodes), dtype=np.int32)
        for i in range(self.num_nodes):
            edges[0, i] = i
            edges[1, i] = (i + 1) % self.num_nodes
        return edges

    def _create_star_edges(self) -> np.ndarray:
        """Create star graph edges (bidirectional)."""
        num_edges = 2 * (self.num_nodes - 1)
        edges = np.zeros((2, num_edges), dtype=np.int32)
        center = 0
        idx = 0
        for i in range(1, self.num_nodes):
            edges[0, idx] = center
            edges[1, idx] = i
            idx += 1
            edges[0, idx] = i
            edges[1, idx] = center
            idx += 1
        return edges

    def _create_complete_edges(self) -> np.ndarray:
        """Create complete graph edges (all directed pairs)."""
        edges_list = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    edges_list.append([i, j])
        return np.array(edges_list, dtype=np.int32).T

    def _create_random_edges(self, rng: np.random.RandomState) -> np.ndarray:
        """Create random directed edges."""
        # Random number of edges (between num_nodes and 2*num_nodes)
        num_edges = rng.randint(self.num_nodes, 2 * self.num_nodes + 1)
        edges = []
        edge_set = set()

        while len(edges) < num_edges:
            src = rng.randint(0, self.num_nodes)
            dst = rng.randint(0, self.num_nodes)
            if src != dst and (src, dst) not in edge_set:
                edges.append([src, dst])
                edge_set.add((src, dst))

        return np.array(edges, dtype=np.int32).T


class NodePredictionGCN(nnx.Module):
    """GNN for node feature prediction (regression)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs = None,
    ):
        self.conv1 = GCNConv(
            in_features, hidden_features, add_self_loops=True, normalize=True, bias=True, rngs=rngs
        )
        self.conv2 = GCNConv(
            hidden_features, out_features, add_self_loops=True, normalize=True, bias=True, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node predictions (num_nodes, out_features)
        """
        x = self.conv1(x, edge_index)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


def compute_metrics(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Compute regression metrics."""
    mse = jnp.mean((y_pred - y_true) ** 2)
    mae = jnp.mean(jnp.abs(y_pred - y_true))

    # Compute R² score
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {"mse": mse, "mae": mae, "r2": r2}


@nnx.jit
def train_step(
    model: NodePredictionGCN, optimizer: nnx.Optimizer, batch: dict[str, jnp.ndarray]
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Single training step with JIT compilation."""

    def loss_fn(model):
        predictions = model(batch["x"], batch["edge_index"])
        loss = jnp.mean((predictions - batch["y"]) ** 2)
        metrics = compute_metrics(predictions, batch["y"])
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    return loss, metrics


@nnx.jit
def eval_step(model: NodePredictionGCN, batch: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Single evaluation step with JIT compilation."""
    predictions = model(batch["x"], batch["edge_index"])
    metrics = compute_metrics(predictions, batch["y"])
    return metrics


def create_dataloader(
    num_samples: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    num_epochs: int = None,
    **data_kwargs,
) -> grain.DataLoader:
    """Create a grain DataLoader with the specified configuration."""

    # Create data source
    dataset = GraphBatchDataSource(num_samples=num_samples, seed=seed, **data_kwargs)

    # Create sampler
    sampler = grain.samplers.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
    )

    # Create DataLoader
    # Note: worker_count=0 runs in main process (no multiprocessing)
    # Set worker_count>0 for parallel data loading in production

    # Use custom batch operation for graph data
    batch_op = grain.transforms.Batch(batch_size=batch_size, drop_remainder=True)
    graph_batch_op = BatchGraphs()

    loader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        worker_count=num_workers,
        operations=[batch_op, graph_batch_op],
    )

    return loader


def main():
    """Main training loop."""

    logging.set_verbosity(logging.INFO)

    logging.info("=" * 60)
    logging.info("Batch Processing with grain.DataLoader")
    logging.info("=" * 60)

    device = jax.devices()[0]
    logging.info(f"\nDevice: {device}")

    # Configuration
    config = {
        "batch_size": 16,
        "num_nodes": 6,
        "feature_dim": 16,
        "hidden_dim": 32,
        "output_dim": 8,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "train_samples": 100,  # Samples per epoch
        "eval_samples": 50,
        "num_workers": 8,
    }

    logging.info("\nConfiguration:")
    logging.info("-" * 40)
    for key, value in config.items():
        logging.info(f"{key}: {value}")

    # Create DataLoaders
    train_loader = create_dataloader(
        num_samples=config["train_samples"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        seed=42,
        num_nodes=config["num_nodes"],
        feature_dim=config["feature_dim"],
        output_dim=config["output_dim"],
    )

    eval_loader = create_dataloader(
        num_samples=config["eval_samples"],
        batch_size=config["batch_size"],
        num_workers=0,  # No workers for eval
        shuffle=False,
        seed=123,
        num_nodes=config["num_nodes"],
        feature_dim=config["feature_dim"],
        output_dim=config["output_dim"],
        num_epochs=1,
    )

    # Create model
    rngs = nnx.Rngs(0)
    model = NodePredictionGCN(
        in_features=config["feature_dim"],
        hidden_features=config["hidden_dim"],
        out_features=config["output_dim"],
        dropout_rate=0.1,
        rngs=rngs,
    )
    model.train()
    eval_model = nnx.merge(*nnx.split(model))
    eval_model.eval()

    optimizer = nnx.Optimizer(model, optax.adam(config["learning_rate"]), wrt=nnx.Param)

    # Training loop
    logging.info("\nTraining:")
    logging.info("-" * 40)
    logging.info("Epoch |  Loss   |   MAE   |   R²    | Time")
    logging.info("-" * 45)

    train_iter = iter(train_loader)

    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()
        model.train()

        # Training
        train_losses = []
        for _ in range(config["train_samples"] // config["batch_size"]):
            batch_np = next(train_iter)
            # Convert NumPy to JAX arrays
            batch_jax = {k: jnp.array(v) for k, v in batch_np.items()}

            loss, _ = train_step(model, optimizer, batch_jax)
            train_losses.append(float(loss))

        # Evaluation every epoch
        model.eval()
        eval_metrics_list = []
        eval_iter = iter(eval_loader)
        for _ in range(config["eval_samples"] // config["batch_size"]):
            batch_np = next(eval_iter)
            batch_jax = {k: jnp.array(v) for k, v in batch_np.items()}
            metrics = eval_step(model, batch_jax)
            eval_metrics_list.append(metrics)

        # Aggregate metrics
        avg_loss = np.mean(train_losses)
        avg_mae = np.mean([float(m["mae"]) for m in eval_metrics_list])
        avg_r2 = np.mean([float(m["r2"]) for m in eval_metrics_list])
        epoch_time = time.time() - epoch_start

        logging.info(
            f"  {epoch:3d} | {avg_loss:7.5f} | {avg_mae:7.5f} | {avg_r2:7.4f} | {epoch_time:.2f}s"
        )

    # Final evaluation
    logging.info("\nFinal Evaluation:")
    logging.info("-" * 40)

    eval_metrics_list = []
    eval_iter = iter(eval_loader)
    for _ in range(config["eval_samples"] // config["batch_size"]):
        batch_np = next(eval_iter)
        batch_jax = {k: jnp.array(v) for k, v in batch_np.items()}
        metrics = eval_step(eval_model, batch_jax)
        eval_metrics_list.append(metrics)

    final_mse = np.mean([float(m["mse"]) for m in eval_metrics_list])
    final_mae = np.mean([float(m["mae"]) for m in eval_metrics_list])
    final_r2 = np.mean([float(m["r2"]) for m in eval_metrics_list])

    logging.info(f"MSE: {final_mse:.6f}")
    logging.info(f"MAE: {final_mae:.6f}")
    logging.info(f"R²:  {final_r2:.6f}")

    logging.info("\n✓ Training completed successfully!")
    logging.info("✓ grain.DataLoader provides efficient CPU-GPU pipeline")
    logging.info("✓ Set worker_count>0 for parallel data loading in production")


if __name__ == "__main__":
    main()
