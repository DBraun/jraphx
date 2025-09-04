# JraphX Examples

This directory contains examples demonstrating how to use JraphX for graph neural network tasks.

## Examples Overview

### Core JraphX Examples

- **`gcn_jraphx.py`** - Graph Convolutional Network using JraphX's built-in layers
  - Demonstrates basic GCN usage with synthetic data
  - Shows data parallelism using `nnx.shard_map` across multiple devices
  - Uses JraphX's `Data` structure for graph representation (node features, edges, labels, etc.)

- **`gcn_standalone.py`** - Standalone GCN implementation from scratch
  - Educational example showing GCN internals without importing JraphX
  - Implements message passing, aggregation, and normalization manually
  - Useful for understanding how GNNs work under the hood

### PyTorch Geometric Integration

- **`karate_club.py`** - Zachary's Karate Club node classification
  - Loads the classic Karate Club dataset from PyTorch Geometric
  - Converts PyG data to JraphX format
  - Semi-supervised learning with only 4 labeled nodes
  - GPU acceleration support

- **`cora_planetoid.py`** - Cora citation network classification
  - Larger graph with 2708 nodes and 10556 edges
  - Proper train/val/test splits for evaluation
  - Achieves ~79% test accuracy, comparable to PyG baselines
  - GPU acceleration support

- **`graph_saint_flickr.py`** - GraphSAINT implementation with Flickr dataset
  - Large-scale graph with 89,250 nodes and ~900K edges
  - Implements random walk sampling for mini-batch training
  - 3-layer GCN with dropout and concatenated representations
  - Includes node/edge weight normalization for unbiased training
  - Suitable for GPU benchmarking and performance comparisons

### Model Examples

- **`pre_built_models.py`** - Comprehensive demonstration of pre-built GNN models
  - Shows usage of GCN, GAT, GraphSAGE, and GIN models
  - Demonstrates various configurations (normalization, dropout, JumpingKnowledge)
  - Includes model comparison on node classification task
  - Semi-supervised learning on Karate Club dataset

### Advanced Examples

- **`batch_node_prediction.py`** - Batch processing with variable edge structures
  - Demonstrates batching graphs with fixed node count (6) but different edges
  - Includes cycle, star, complete, and random graph patterns
  - Node feature prediction (regression) task with MSE loss
  - Full JIT compilation for training and evaluation
  - Shows how to handle heterogeneous graph structures in batches
  - Uses Google's grain library for high-performance data loading
  - Implements RandomAccessDataSource with pure NumPy for CPU operations
  - Custom graph batching operation for combining multiple graphs
  - Multi-worker parallel data loading for CPU-GPU pipeline

- **`gat_example.py`** - Graph Attention Networks (GAT and GATv2)
  - Comprehensive demonstration of GAT and GATv2 layers
  - Shows differences between GAT and GATv2 attention mechanisms
  - Edge feature handling for attention computation
  - Bipartite graph support (e.g., user-item interactions)
  - Residual connections for deeper networks
  - Attention weight visualization
  - Performance benchmarks comparing different configurations
  - JIT compilation for optimal performance

- **`nnx_transforms.py`** - Flax NNX transformations with JraphX
  - Demonstrates NNX's `vmap`, `scan`, `jit`, and `grad` transformations
  - Example 1: Parallel batch processing with `nnx.vmap`
  - Example 2: Sequential GNN layers using `nnx.scan` for layer stacking
  - Example 3: Temporal Graph Networks with two approaches:
    - Custom `GraphGRUCell` following NNX RNNCellBase pattern
    - Integration of NNX's built-in `GRUCell` with graph preprocessing
    - Proper `initialize_carry` method for state management
    - Optimized gate computations with combined Linear layers
  - Example 4: Memory-efficient training with `scan` over mini-batches
  - Uses `nnx.Optimizer` with Adam optimization
  - Model architecture visualization with `nnx.tabulate(depth=1)`
  - Proper use of `nnx.Carry` and `StateAxes` for stateful transformations
  - Shows integration of NNX RNN cells with graph neural networks


- **`scatter_operations_demo.py`** - Scatter operations for GNNs
  - Demonstrates all scatter operations including new additions:
    - `scatter_std`: Standard deviation aggregation
    - `scatter_logsumexp`: Numerically stable log-sum-exp
    - `scatter_softmax`: Attention weight normalization
    - `scatter_log_softmax`: Log-space softmax for stability
  - Shows practical use cases in graph pooling and attention
  - Numerical stability demonstrations

### Benchmarking

- **`graph_saint_flickr_benchmark.py`** - Performance comparison between PyG and JraphX
  - Runs GraphSAINT implementations on both frameworks
  - Measures training speed (epochs/sec, samples/sec)
  - Compares memory usage and final accuracy
  - Provides side-by-side performance metrics for GPU workloads

### Dependencies

For PyTorch Geometric examples, install the additional dependencies:

```bash
pip install torch-geometric
```

Or in the root of jraphx, install all example dependencies:

```bash
pip install -e ".[examples]"
```

## Performance Notes

- The examples automatically detect and use GPU if available
- Synthetic data examples use CPU with data parallelism across multiple devices
- Real dataset examples (Karate Club, Cora) benefit significantly from GPU acceleration
- Memory-efficient examples are crucial for large-scale graph learning

## Learn More

- See the [main README](../README.md) for JraphX installation and overview
- Check individual example files for detailed comments and explanations
- Refer to the [documentation](../docs/) for API details
