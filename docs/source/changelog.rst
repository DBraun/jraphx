Changelog
=========

Version 0.0.1
-------------

Initial release of JraphX.

Features
~~~~~~~~

**Core Data Structures**

* ``Data`` class: Single graph representation with node features, edge indices, edge attributes, and graph-level properties
* ``Batch`` class: Efficient batching of multiple graphs into disconnected graph batches with automatic indexing management

**Message Passing Framework**

* Unified ``MessagePassing`` base class providing a standardized interface for all graph neural network layers
* Flexible message computation, aggregation (sum, mean, max, min), and node update functions
* Support for both node-to-node and edge-enhanced message passing paradigms

**Graph Convolution Layers**

* ``GCNConv``: Graph Convolutional Network with spectral-based convolution and optional edge weights
* ``GATConv``: Graph Attention Network with multi-head attention mechanism and learnable attention weights
* ``GATv2Conv``: Improved Graph Attention Network with enhanced attention computation for better expressivity
* ``GraphSAGE`` (``SAGEConv``): GraphSAGE with multiple aggregation functions (mean, max, LSTM) for inductive learning
* ``GINConv``: Graph Isomorphism Network with theoretical guarantees for graph representation power
* ``EdgeConv``: Dynamic edge convolution for learning on point clouds and dynamic graph construction
* ``DynamicEdgeConv``: Enhanced EdgeConv with k-nearest neighbor graph construction
* ``TransformerConv``: Graph Transformer layer with optimized query-key-value projections and positional encodings

**Pooling Operations**

* **Global pooling**: ``global_add_pool``, ``global_mean_pool``, ``global_max_pool``, ``global_min_pool`` for graph-level representations
* **Advanced pooling**: ``global_softmax_pool``, ``global_sort_pool`` for differentiable and sorted aggregations
* **Hierarchical pooling**: ``TopKPooling`` and ``SAGPooling`` for coarsening graph structures with learnable node selection
* **Batched operations**: Optimized versions (``batched_global_*_pool``) for efficient parallel processing of graph batches

**Utility Functions**

* **Scatter operations**: Comprehensive set including ``scatter_add``, ``scatter_mean``, ``scatter_max``, ``scatter_min``, ``scatter_std``, ``scatter_logsumexp`` for flexible aggregation
* **Scatter softmax**: ``scatter_softmax``, ``scatter_log_softmax``, ``masked_scatter_softmax`` for attention-like mechanisms
* **Graph utilities**: Degree computation (``degree``, ``in_degree``, ``out_degree``), self-loop management (``add_self_loops``, ``remove_self_loops``)
* **Conversion functions**: ``to_dense_adj``, ``to_edge_index``, ``to_undirected`` for different graph representations
* **Graph preprocessing**: ``coalesce`` for edge deduplication, ``maybe_num_nodes`` for automatic node count inference

**Pre-built Models**

* ``GCN``, ``GAT``, ``GraphSAGE``, ``GIN``: Complete model implementations with configurable depth, hidden dimensions, and activation functions
* ``JumpingKnowledge``: Multi-layer aggregation with concatenation, max, and LSTM-based combination strategies
* ``MLP``: Multi-layer perceptron with dropout, batch normalization, and flexible activation functions
* ``BasicGNN``: Abstract base class for implementing custom GNN architectures with standardized interfaces

**Normalization Layers**

* ``BatchNorm``: Batch normalization with running statistics for stable training across graph batches
* ``LayerNorm``: Layer normalization supporting both node-wise and graph-wise normalization schemes
* ``GraphNorm``: Graph-specific normalization designed for graph neural network architectures

**JAX Integration & Performance**

* Extensive use of ``jax.vmap`` and ``nnx.vmap`` for efficient parallel processing of graph batches
* Memory-efficient training patterns using ``jax.lax.scan`` and ``nnx.scan`` for sequential operations
* JIT compilation support for all operations with optimized JAX primitives
* Native integration with Flax NNX for modern JAX neural network development
* Efficient scatter operations using JAX's advanced indexing (``at[].add/max/min``) for high-performance aggregation
