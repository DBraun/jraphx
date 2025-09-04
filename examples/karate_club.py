"""
Zachary's Karate Club Example using JraphX with PyTorch Geometric Dataset
=========================================================================

TODO: Data conversion from PyTorch to JAX adds overhead
TODO: PyG's train_mask uses only 4 labeled nodes (one per class) which is very sparse
TODO: Consider adding validation split for better evaluation

This example demonstrates:
1. Loading Zachary's Karate Club dataset from PyTorch Geometric
2. Converting PyG data to jraphx format
3. Training a GCN for node classification using jraphx
4. Evaluating on the full graph

The Karate Club dataset represents a social network of 34 members of a
karate club, with edges representing interactions between members. The task
is to predict which faction each member belongs to after a split in the club.

See https://en.wikipedia.org/wiki/Zachary%27s_karate_club.
"""

import jax
import optax
from flax import nnx
from flax.struct import dataclass
from jax import numpy as jnp

# Import jraphx components
import jraphx
from jraphx.nn.conv import GCNConv


@dataclass
class Data(jraphx.Data):
    train_mask: jnp.ndarray | None = None


# Import PyTorch Geometric for dataset
try:
    from torch_geometric.datasets import KarateClub as PyGKarateClub
except ImportError as err:
    raise RuntimeError(
        "PyTorch Geometric is required for this example. "
        "Install it with: pip install torch-geometric"
    ) from err


def pyg_to_jraphx(pyg_data):
    """
    Convert PyTorch Geometric Data to jraphx Data format.

    Args:
        pyg_data: PyTorch Geometric Data object

    Returns:
        jraphx Data object with JAX arrays
    """
    # Convert PyTorch tensors to JAX arrays
    x = jnp.array(pyg_data.x.numpy()) if pyg_data.x is not None else None
    edge_index = jnp.array(pyg_data.edge_index.numpy())
    y = jnp.array(pyg_data.y.numpy()) if pyg_data.y is not None else None

    if hasattr(pyg_data, "train_mask"):
        train_mask = jnp.array(pyg_data.train_mask.numpy())
    else:
        train_mask = None

    # Create jraphx Data object
    jraphx_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

    return jraphx_data


class KarateGCN(nnx.Module):
    """
    2-layer GCN for Karate Club node classification.
    """

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, *, rngs: nnx.Rngs):
        self.conv1 = GCNConv(
            in_features, hidden_dim, normalize=True, add_self_loops=True, bias=True, rngs=rngs
        )
        self.conv2 = GCNConv(
            hidden_dim, num_classes, normalize=True, add_self_loops=True, bias=True, rngs=rngs
        )
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        # First layer
        x = self.conv1(x, edge_index)
        x = nnx.relu(x)
        x = self.dropout(x)

        # Second layer
        x = self.conv2(x, edge_index)

        return x


def train_step(model: KarateGCN, optimizer: nnx.Optimizer, data: Data, train_mask: jnp.ndarray):
    """Training step with masked loss."""
    model.train()  # Set model to training mode

    @nnx.value_and_grad
    def loss_fn(model, data, mask):
        logits = model(data.x, data.edge_index)
        # Only compute loss on training nodes
        train_logits = logits[mask]
        train_labels = data.y[mask]
        loss = optax.softmax_cross_entropy_with_integer_labels(train_logits, train_labels)
        return loss.mean()

    loss, grads = loss_fn(model, data, train_mask)
    optimizer.update(model, grads)

    return loss


def evaluate(model: KarateGCN, data: Data, mask: jnp.ndarray = None):
    """Evaluate model accuracy."""
    model.eval()

    logits = model(data.x, data.edge_index)
    predictions = jnp.argmax(logits, axis=-1)

    if mask is not None:
        # Evaluate only on masked nodes
        correct = (predictions[mask] == data.y[mask]).sum()
        total = mask.sum()
    else:
        # Evaluate on all nodes
        correct = (predictions == data.y).sum()
        total = len(data.y)

    accuracy = correct / total
    return float(accuracy)


def visualize_predictions(model: KarateGCN, data: Data):
    """Print prediction statistics."""
    model.eval()

    logits = model(data.x, data.edge_index)
    predictions = jnp.argmax(logits, axis=-1)
    probs = nnx.softmax(logits, axis=-1)

    print("\nNode Classification Results:")
    print("-" * 50)

    # Count predictions per class
    unique_preds, counts = jnp.unique(predictions, return_counts=True)
    print("\nPredicted class distribution:")
    for class_id, count in zip(unique_preds, counts, strict=False):
        print(f"  Class {class_id}: {count} nodes")

    # Show true class distribution
    unique_true, true_counts = jnp.unique(data.y, return_counts=True)
    print("\nTrue class distribution:")
    for class_id, count in zip(unique_true, true_counts, strict=False):
        print(f"  Class {class_id}: {count} nodes")

    # Show confidence statistics
    max_probs = jnp.max(probs, axis=-1)
    print("\nConfidence statistics:")
    print(f"  Mean confidence: {max_probs.mean():.3f}")
    print(f"  Min confidence: {max_probs.min():.3f}")
    print(f"  Max confidence: {max_probs.max():.3f}")

    # Show misclassified nodes
    misclassified = jnp.where(predictions != data.y)[0]
    if len(misclassified) > 0:
        print(f"\nMisclassified nodes: {len(misclassified)}/{data.num_nodes}")
        if len(misclassified) <= 10:
            for node_id in misclassified[:10]:
                print(f"  Node {node_id}: predicted {predictions[node_id]}, true {data.y[node_id]}")


def main():
    """Train GCN on Karate Club dataset."""

    print("=" * 60)
    print("Zachary's Karate Club - Node Classification with JraphX")
    print("=" * 60)

    # Check for GPU availability
    devices = jax.devices()
    device = devices[0]
    print(f"\nUsing device: {device}")
    print(f"Device type: {'GPU' if 'cuda' in str(device).lower() else 'CPU'}")

    # Load dataset from PyTorch Geometric
    print("\n1. Loading Dataset from PyTorch Geometric")
    print("-" * 40)

    pyg_dataset = PyGKarateClub()
    pyg_data = pyg_dataset[0]

    print(f"Dataset: {pyg_dataset}")
    print(f"Number of graphs: {len(pyg_dataset)}")
    print(f"Number of features: {pyg_dataset.num_features}")
    print(f"Number of classes: {pyg_dataset.num_classes}")

    # Convert to jraphx format
    print("\n2. Converting to JraphX Format")
    print("-" * 40)

    data = pyg_to_jraphx(pyg_data)

    # Move data to GPU if available
    if "cuda" in str(device).lower():
        data = jax.device_put(data, device)
        print("Data moved to GPU")

    train_mask = data.train_mask

    print(f"JraphX Data: {data}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Labels shape: {data.y.shape}")
    print(f"Training nodes: {train_mask.sum()}/{data.num_nodes}")

    # Create model
    print("\n3. Creating Model")
    print("-" * 40)

    rngs = nnx.Rngs(42)

    model = KarateGCN(
        in_features=pyg_dataset.num_features,
        hidden_dim=16,
        num_classes=pyg_dataset.num_classes,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    print("Model created with hidden dimension: 16")
    print(f"Input features: {pyg_dataset.num_features}")
    print(f"Output classes: {pyg_dataset.num_classes}")

    # Training
    print("\n4. Training")
    print("-" * 40)

    num_epochs = 200
    model.train()

    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, data, train_mask)

        if epoch % 20 == 0:
            train_acc = evaluate(model, data, train_mask)
            full_acc = evaluate(model, data)
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train Acc: {train_acc:.3f} | Full Acc: {full_acc:.3f}"
            )

    # Final evaluation
    print("\n5. Final Evaluation")
    print("-" * 40)

    train_accuracy = evaluate(model, data, train_mask)
    full_accuracy = evaluate(model, data)

    print(f"Final training accuracy: {train_accuracy:.3f}")
    print(f"Final full graph accuracy: {full_accuracy:.3f}")

    # Visualize predictions
    visualize_predictions(model, data)

    # Analysis
    print("\n6. Analysis")
    print("-" * 40)
    print("Note: The Karate Club dataset has only 4 labeled training nodes")
    print("(one per class), making it a challenging semi-supervised task.")
    print("The model must propagate information from these few labeled")
    print("nodes to classify the entire graph.")

    print("\nComparison with PyTorch Geometric:")
    print("- Data conversion from PyTorch tensors to JAX arrays works smoothly")
    print("- jraphx's GCNConv provides similar functionality to PyG's version")
    print("- Training is efficient even on small graphs like Karate Club")

    print("\nPotential improvements:")
    print("- Add validation split for better model selection")
    print("- Try different GNN architectures (GAT, GraphSAGE)")
    print("- Experiment with different numbers of labeled nodes")


if __name__ == "__main__":
    main()
