"""
GraphSAINT implementation with Flickr dataset for jraphx.

This example implements GraphSAINT random walk sampling for scalable
training on the Flickr dataset (89,250 nodes, ~899K edges).
"""

# Add parent directory to path
import os
import time

import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax import nnx
from flax.struct import dataclass
from jax import random
from torch_geometric.datasets import Flickr

from jraphx import Data
from jraphx.nn.conv import GCNConv
from jraphx.nn.norm import BatchNorm
from jraphx.utils import degree


@dataclass
class FlickrData(Data):
    """Extended Data class for Flickr dataset with additional fields."""

    edge_weight: jnp.ndarray | None = None
    train_mask: jnp.ndarray | None = None
    val_mask: jnp.ndarray | None = None
    test_mask: jnp.ndarray | None = None
    num_classes: int | None = None


def load_flickr_dataset(path: str | None = None):
    """Load Flickr dataset from PyTorch Geometric."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "Flickr")

    dataset = Flickr(path)
    pyg_data = dataset[0]

    # Convert to jraphx format
    x = jnp.array(pyg_data.x.numpy())
    edge_index = jnp.array(pyg_data.edge_index.numpy())
    y = jnp.array(pyg_data.y.numpy())

    # Create masks
    train_mask = jnp.array(pyg_data.train_mask.numpy())
    val_mask = jnp.array(pyg_data.val_mask.numpy())
    test_mask = jnp.array(pyg_data.test_mask.numpy())

    # Compute edge weights (normalized by in-degree)
    col = edge_index[1]
    in_degrees = degree(col, num_nodes=x.shape[0])
    edge_weight = 1.0 / in_degrees[col]

    data = FlickrData(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=dataset.num_classes,
    )

    return data, dataset.num_features, dataset.num_classes


def random_walk_sampling(
    edge_index: jnp.ndarray,
    num_nodes: int,
    batch_size: int,
    walk_length: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform random walk sampling for GraphSAINT.

    Returns sampled node indices and their sampling probabilities.
    """
    # Start nodes for random walks
    key1, key2 = random.split(key)
    start_nodes = random.choice(key1, num_nodes, shape=(batch_size,), replace=True)

    # Build adjacency list representation for efficient sampling
    # For simplicity, we'll do a basic random walk
    sampled_nodes = start_nodes
    current_nodes = start_nodes

    for _ in range(walk_length):
        # Get neighbors of current nodes
        mask = jnp.isin(edge_index[0], current_nodes)
        available_edges = edge_index[:, mask]

        if available_edges.shape[1] > 0:
            # Sample next nodes
            key2, key3 = random.split(key2)
            idx = random.choice(
                key3, available_edges.shape[1], shape=(min(batch_size, available_edges.shape[1]),)
            )
            next_nodes = available_edges[1, idx]
            current_nodes = next_nodes
            sampled_nodes = jnp.concatenate([sampled_nodes, next_nodes])

    # Remove duplicates
    sampled_nodes = jnp.unique(sampled_nodes)

    # Compute sampling probabilities (uniform for simplicity)
    sampling_probs = jnp.ones(sampled_nodes.shape[0]) / num_nodes

    return sampled_nodes, sampling_probs


def extract_subgraph(
    data: FlickrData, node_idx: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract subgraph for sampled nodes."""
    # Create node mapping
    node_mask = jnp.zeros(data.x.shape[0], dtype=bool)
    node_mask = node_mask.at[node_idx].set(True)

    # Extract edges within sampled nodes
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    sub_edge_index = data.edge_index[:, edge_mask]
    sub_edge_weight = data.edge_weight[edge_mask]

    # Remap node indices to 0...n-1
    node_mapping = jnp.cumsum(node_mask) - 1
    sub_edge_index = node_mapping[sub_edge_index]

    # Extract features and labels
    sub_x = data.x[node_idx]
    sub_y = data.y[node_idx]
    sub_train_mask = data.train_mask[node_idx]

    return sub_x, sub_edge_index, sub_edge_weight, sub_y, sub_train_mask, node_idx


class GraphSAINTModel(nnx.Module):
    """3-layer GCN model for GraphSAINT with optional batch normalization."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv1 = GCNConv(in_features, hidden_features, rngs=rngs)
        self.conv2 = GCNConv(hidden_features, hidden_features, rngs=rngs)
        self.conv3 = GCNConv(hidden_features, hidden_features, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # Add batch normalization layers if requested
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = BatchNorm(hidden_features, momentum=0.9)
            self.bn2 = BatchNorm(hidden_features, momentum=0.9)
            self.bn3 = BatchNorm(hidden_features, momentum=0.9)

        # Final linear layer for classification
        self.classifier = nnx.Linear(3 * hidden_features, out_features, rngs=rngs)
        self.rngs = rngs

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # First layer
        x1 = self.conv1(x, edge_index, edge_weight)
        if self.use_batch_norm:
            x1 = self.bn1(x1)
        x1 = nnx.relu(x1)
        x1 = self.dropout(x1)

        # Second layer
        x2 = self.conv2(x1, edge_index, edge_weight)
        if self.use_batch_norm:
            x2 = self.bn2(x2)
        x2 = nnx.relu(x2)
        x2 = self.dropout(x2)

        # Third layer
        x3 = self.conv3(x2, edge_index, edge_weight)
        if self.use_batch_norm:
            x3 = self.bn3(x3)
        x3 = nnx.relu(x3)
        x3 = self.dropout(x3)

        # Concatenate all representations
        x = jnp.concatenate([x1, x2, x3], axis=-1)

        # Final classification
        return self.classifier(x)


@jax.jit
def compute_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
    node_norm: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute cross-entropy loss with optional node normalization."""
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    if node_norm is not None:
        loss = loss * node_norm

    # Apply mask and compute mean
    masked_loss = jnp.where(mask, loss, 0.0)
    return jnp.sum(masked_loss) / jnp.sum(mask)


@nnx.jit
def evaluate_jit(
    model: GraphSAINTModel,
    x: jnp.ndarray,
    edge_index: jnp.ndarray,
    edge_weight: jnp.ndarray,
    y: jnp.ndarray,
    train_mask: jnp.ndarray,
    val_mask: jnp.ndarray,
    test_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation on full graph."""
    # Model should already be in eval mode
    logits = model(x, edge_index, edge_weight)
    predictions = jnp.argmax(logits, axis=-1)

    # Compute accuracies using where to avoid boolean indexing
    train_correct = jnp.where(train_mask, predictions == y, False)
    val_correct = jnp.where(val_mask, predictions == y, False)
    test_correct = jnp.where(test_mask, predictions == y, False)

    train_acc = jnp.sum(train_correct) / jnp.sum(train_mask)
    val_acc = jnp.sum(val_correct) / jnp.sum(val_mask)
    test_acc = jnp.sum(test_correct) / jnp.sum(test_mask)

    return train_acc, val_acc, test_acc


def evaluate(model: GraphSAINTModel, data: FlickrData) -> tuple[float, float, float]:
    """Evaluate model on full graph."""
    model.eval()  # Set model to evaluation mode
    train_acc, val_acc, test_acc = evaluate_jit(
        model,
        data.x,
        data.edge_index,
        data.edge_weight,
        data.y,
        data.train_mask,
        data.val_mask,
        data.test_mask,
    )
    return float(train_acc), float(val_acc), float(test_acc)


@nnx.jit
def train_step_jit(
    model: GraphSAINTModel,
    optimizer,
    sub_x: jnp.ndarray,
    sub_edge_index: jnp.ndarray,
    sub_edge_weight: jnp.ndarray,
    sub_y: jnp.ndarray,
    sub_train_mask: jnp.ndarray,
    node_norm: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled training step."""

    def loss_fn(model):
        # Model should already be in train mode
        logits = model(sub_x, sub_edge_index, sub_edge_weight)
        return compute_loss(logits, sub_y, sub_train_mask, node_norm)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    batch_samples = jnp.sum(sub_train_mask)
    return loss, batch_samples


def train_epoch(
    model: GraphSAINTModel,
    optimizer,
    data: FlickrData,
    batch_size: int,
    walk_length: int,
    num_steps: int,
    key: jax.Array,
    use_normalization: bool = False,
) -> float:
    """Train for one epoch with GraphSAINT sampling."""
    model.train()  # Set model to training mode
    total_loss = 0.0
    total_samples = 0

    for _ in range(num_steps):
        key, subkey = random.split(key)

        # Sample subgraph
        sampled_nodes, sampling_probs = random_walk_sampling(
            data.edge_index, data.x.shape[0], batch_size, walk_length, subkey
        )

        # Extract subgraph
        sub_x, sub_edge_index, sub_edge_weight, sub_y, sub_train_mask, node_idx = extract_subgraph(
            data, sampled_nodes
        )

        # Compute node normalization if needed
        node_norm = None
        if use_normalization:
            node_norm = 1.0 / (sampling_probs * data.x.shape[0])

        # JIT-compiled training step
        loss, batch_samples = train_step_jit(
            model,
            optimizer,
            sub_x,
            sub_edge_index,
            sub_edge_weight,
            sub_y,
            sub_train_mask,
            node_norm,
        )

        total_loss += float(loss) * float(batch_samples)
        total_samples += float(batch_samples)

    return total_loss / max(total_samples, 1)


def main():
    """Main training loop."""
    # Set random seed
    key = random.key(42)

    logging.set_verbosity(logging.INFO)

    # Load dataset
    logging.info("Loading Flickr dataset...")
    data, in_features, out_features = load_flickr_dataset()
    logging.info(f"Dataset: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    logging.info(f"Features: {in_features}, Classes: {out_features}")

    # Initialize model
    hidden_features = 256
    dropout_rate = 0.2

    rngs = nnx.Rngs(0)
    model = GraphSAINTModel(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )

    # Initialize optimizer (using nnx.Optimizer with wrt)
    learning_rate = 0.001
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Training parameters
    num_epochs = 50
    batch_size = 6000
    walk_length = 2
    num_steps_per_epoch = 5
    use_normalization = False  # Set to True for normalized training

    # Check for GPU
    device = jax.devices()[0]
    logging.info(f"Using device: {device}")

    # Training loop
    logging.info("Starting training...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        key, subkey = random.split(key)
        train_loss = train_epoch(
            model,
            optimizer,
            data,
            batch_size,
            walk_length,
            num_steps_per_epoch,
            subkey,
            use_normalization,
        )

        # Evaluate
        train_acc, val_acc, test_acc = evaluate(model, data)

        epoch_time = time.time() - start_time

        logging.info(
            f"Epoch {epoch:03d}: Loss: {train_loss:.4f}, "
            f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

    logging.info("Training completed!")

    # Final evaluation
    train_acc, val_acc, test_acc = evaluate(model, data)
    logging.info(
        f"Final Results - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
