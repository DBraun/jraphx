"""
Benchmark comparison between PyTorch Geometric and jraphx GraphSAINT implementations.

This script runs both implementations and compares:
- Training speed (epochs/sec, samples/sec)
- Memory usage
- Final accuracy
"""

import argparse
import gc
import os
import time
from typing import Any

import jax
import numpy as np
import optax
import torch
import torch.nn.functional as F
from absl import logging
from flax import nnx
from graph_saint_flickr import GraphSAINTModel, evaluate, load_flickr_dataset, train_epoch
from jax import random
from torch_geometric.datasets import Flickr
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree


def run_pytorch_geometric(num_epochs: int = 10, batch_size: int = 6000) -> dict[str, Any]:
    """Run PyTorch Geometric GraphSAINT implementation."""

    # Load dataset
    path = os.path.join(os.path.dirname(__file__), "..", "data", "Flickr")
    dataset = Flickr(path)
    data = dataset[0]
    row, col = data.edge_index
    data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]

    # Create data loader
    loader = GraphSAINTRandomWalkSampler(
        data,
        batch_size=batch_size,
        walk_length=2,
        num_steps=5,
        sample_coverage=100,
        save_dir=dataset.processed_dir,
        num_workers=0,  # Single thread for fair comparison
    )

    # Define model
    class Net(torch.nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = GraphConv(in_features, hidden_features)
            self.conv2 = GraphConv(hidden_features, hidden_features)
            self.conv3 = GraphConv(hidden_features, hidden_features)
            self.lin = torch.nn.Linear(3 * hidden_features, out_features)
            self.dropout = torch.nn.Dropout(0.2)

        def forward(self, x, edge_index, edge_weight=None):
            x1 = F.relu(self.conv1(x, edge_index, edge_weight))
            x1 = self.dropout(x1)
            x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
            x2 = self.dropout(x2)
            x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
            x3 = self.dropout(x3)
            x = torch.cat([x1, x2, x3], dim=-1)
            return self.lin(x)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(dataset.num_node_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training metrics
    epoch_times = []
    samples_processed = []
    train_losses = []

    # Training loop
    logging.info("Starting PyTorch Geometric training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        start_time = time.time()

        total_loss = 0
        total_samples = 0

        for batch_data in loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            out = model(batch_data.x, batch_data.edge_index, batch_data.edge_weight)
            loss = F.cross_entropy(out[batch_data.train_mask], batch_data.y[batch_data.train_mask])

            loss.backward()
            optimizer.step()

            batch_samples = batch_data.train_mask.sum().item()
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        samples_processed.append(total_samples)
        train_losses.append(total_loss / total_samples)

        logging.info(
            f"PyG Epoch {epoch:02d}: Loss: {train_losses[-1]:.4f}, Time: {epoch_time:.2f}s"
        )

    # Evaluation
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)

        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    # Memory usage (approximate)
    if device.type == "cuda":
        torch.cuda.synchronize()
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0

    return {
        "framework": "PyTorch Geometric",
        "epoch_times": epoch_times,
        "samples_processed": samples_processed,
        "train_losses": train_losses,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "memory_mb": memory_mb,
        "device": str(device),
    }


def run_jraphx(num_epochs: int = 10, batch_size: int = 6000) -> dict[str, Any]:
    """Run jraphx GraphSAINT implementation."""

    # Load dataset
    data, in_features, out_features = load_flickr_dataset()

    # Initialize model
    rngs = nnx.Rngs(0)
    model = GraphSAINTModel(
        in_features=in_features,
        hidden_features=256,
        out_features=out_features,
        dropout_rate=0.2,
        rngs=rngs,
    )

    # Initialize optimizer
    optimizer = nnx.Optimizer(model, optax.adam(0.001), wrt=nnx.Param)

    # Training parameters
    walk_length = 2
    num_steps_per_epoch = 5

    # Training metrics
    epoch_times = []
    samples_processed = []
    train_losses = []

    # Get device info
    device = jax.devices()[0]

    # Training loop
    logging.info("Starting jraphx training...")
    key = random.key(42)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        key, subkey = random.split(key)
        train_loss = train_epoch(
            model,
            optimizer,
            data,
            batch_size,
            walk_length,
            num_steps_per_epoch,
            subkey,
            use_normalization=False,
        )

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        # Approximate samples processed (batch_size * num_steps)
        samples_processed.append(batch_size * num_steps_per_epoch)
        train_losses.append(train_loss)

        logging.info(f"jraphx Epoch {epoch:02d}: Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")

    # Evaluation
    train_acc, val_acc, test_acc = evaluate(model, data)

    # Memory usage (approximate for JAX)
    # JAX doesn't provide easy memory tracking like PyTorch
    memory_mb = 0  # Placeholder

    return {
        "framework": "jraphx",
        "epoch_times": epoch_times,
        "samples_processed": samples_processed,
        "train_losses": train_losses,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "memory_mb": memory_mb,
        "device": str(device),
    }


def print_comparison(pyg_results: dict[str, Any], jraphx_results: dict[str, Any]):
    """Print benchmark comparison results."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON: GraphSAINT on Flickr Dataset")
    print("=" * 60)

    # Device info
    print(f"\nPyTorch Geometric device: {pyg_results['device']}")
    print(f"jraphx device: {jraphx_results['device']}")

    # Training speed
    pyg_avg_time = np.mean(pyg_results["epoch_times"])
    jraphx_avg_time = np.mean(jraphx_results["epoch_times"])
    pyg_samples_per_sec = np.mean(
        [
            s / t
            for s, t in zip(
                pyg_results["samples_processed"], pyg_results["epoch_times"], strict=False
            )
        ]
    )
    jraphx_samples_per_sec = np.mean(
        [
            s / t
            for s, t in zip(
                jraphx_results["samples_processed"], jraphx_results["epoch_times"], strict=False
            )
        ]
    )

    print("\nTraining Speed:")
    print("  PyTorch Geometric:")
    print(f"    - Avg epoch time: {pyg_avg_time:.2f}s")
    print(f"    - Samples/sec: {pyg_samples_per_sec:.0f}")
    print("  jraphx:")
    print(f"    - Avg epoch time: {jraphx_avg_time:.2f}s")
    print(f"    - Samples/sec: {jraphx_samples_per_sec:.0f}")

    # Speed comparison
    speedup = pyg_avg_time / jraphx_avg_time
    if speedup > 1:
        print(f"\n  → jraphx is {speedup:.2f}x faster")
    else:
        print(f"\n  → PyTorch Geometric is {1/speedup:.2f}x faster")

    # Memory usage
    if pyg_results["memory_mb"] > 0:
        print("\nMemory Usage:")
        print(f"  PyTorch Geometric: {pyg_results['memory_mb']:.1f} MB")
        if jraphx_results["memory_mb"] > 0:
            print(f"  jraphx: {jraphx_results['memory_mb']:.1f} MB")

    # Final accuracy
    print("\nFinal Accuracy:")
    print("  PyTorch Geometric:")
    print(f"    - Train: {pyg_results['train_acc']:.4f}")
    print(f"    - Val: {pyg_results['val_acc']:.4f}")
    print(f"    - Test: {pyg_results['test_acc']:.4f}")
    print("  jraphx:")
    print(f"    - Train: {jraphx_results['train_acc']:.4f}")
    print(f"    - Val: {jraphx_results['val_acc']:.4f}")
    print(f"    - Test: {jraphx_results['test_acc']:.4f}")

    # Training curves
    print("\nTraining Loss (first 5 epochs):")
    for i in range(min(5, len(pyg_results["train_losses"]))):
        print(f"  Epoch {i+1}:")
        print(f"    PyG: {pyg_results['train_losses'][i]:.4f}")
        print(f"    jraphx: {jraphx_results['train_losses'][i]:.4f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GraphSAINT: PyG vs jraphx")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=6000, help="Batch size for sampling (default: 6000)"
    )
    parser.add_argument("--pyg-only", action="store_true", help="Run only PyTorch Geometric")
    parser.add_argument("--jraphx-only", action="store_true", help="Run only jraphx")
    args = parser.parse_args()

    results = {}

    # Run PyTorch Geometric
    if not args.jraphx_only:
        try:
            logging.info("Running PyTorch Geometric benchmark...")
            pyg_results = run_pytorch_geometric(args.epochs, args.batch_size)
            results["pyg"] = pyg_results
            gc.collect()
        except Exception as e:
            logging.error(f"PyTorch Geometric benchmark failed: {e}")
            results["pyg"] = None

    # Run jraphx
    if not args.pyg_only:
        try:
            logging.info("Running jraphx benchmark...")
            jraphx_results = run_jraphx(args.epochs, args.batch_size)
            results["jraphx"] = jraphx_results
            gc.collect()
        except Exception as e:
            logging.error(f"jraphx benchmark failed: {e}")
            results["jraphx"] = None

    # Print comparison if both ran successfully
    if "pyg" in results and "jraphx" in results:
        if results["pyg"] is not None and results["jraphx"] is not None:
            print_comparison(results["pyg"], results["jraphx"])
    elif "pyg" in results and results["pyg"] is not None:
        print("\nPyTorch Geometric Results:")
        print(f"  Final Test Accuracy: {results['pyg']['test_acc']:.4f}")
        print(f"  Avg Epoch Time: {np.mean(results['pyg']['epoch_times']):.2f}s")
    elif "jraphx" in results and results["jraphx"] is not None:
        print("\njraphx Results:")
        print(f"  Final Test Accuracy: {results['jraphx']['test_acc']:.4f}")
        print(f"  Avg Epoch Time: {np.mean(results['jraphx']['epoch_times']):.2f}s")


if __name__ == "__main__":
    main()
