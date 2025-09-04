jraphx.nn
=========

.. currentmodule:: jraphx.nn

This module contains neural network layers and operations for graph neural networks.

.. toctree::
   :maxdepth: 2
   :caption: Submodules

   conv
   models
   norm
   pooling

Overview
--------

The ``jraphx.nn`` module provides a comprehensive set of neural network components for building graph neural networks:

**Core Components:**

- **Message Passing Framework** (:doc:`conv`): Base class and implementations for graph convolutions
- **Pre-built Models** (:doc:`models`): Ready-to-use GNN architectures (GCN, GAT, GraphSAGE, GIN)
- **Normalization Layers** (:doc:`norm`): BatchNorm, LayerNorm, and GraphNorm for GNNs
- **Pooling Operations** (:doc:`pooling`): Global and hierarchical pooling methods

Quick Start
-----------

Using Pre-built Models
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.nn.models import GCN
   import flax.nnx as nnx

   # Create a 3-layer GCN model
   model = GCN(
       in_features=16,
       hidden_features=64,
       num_layers=3,
       out_features=10,
       dropout_rate=0.5,
       norm="layer_norm",
       rngs=nnx.Rngs(0)
   )

   # Forward pass
   out = model(x, edge_index, batch=batch)

Building Custom Models
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.nn.conv import GCNConv, GATConv
   from jraphx.nn.norm import GraphNorm
   from jraphx.nn.pool import TopKPooling, global_mean_pool
   import flax.nnx as nnx

   class CustomGNN(nnx.Module):
       def __init__(self, in_features, out_features, rngs):
           self.conv1 = GCNConv(in_features, 64, rngs=rngs)
           self.norm1 = GraphNorm(64, rngs=rngs)
           self.pool1 = TopKPooling(64, ratio=0.8, rngs=rngs)

           self.conv2 = GATConv(64, 64, heads=4, rngs=rngs)
           self.norm2 = GraphNorm(256, rngs=rngs)  # 64 * 4 heads

           self.classifier = nnx.Linear(256, out_features, rngs=rngs)
           self.dropout = nnx.Dropout(0.5, rngs=rngs)

       def __call__(self, x, edge_index, batch):
           # First conv block
           x = self.conv1(x, edge_index)
           x = self.norm1(x, batch)
           x = nnx.relu(x)

           # Pooling
           x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)

           # Second conv block (GAT)
           x = self.conv2(x, edge_index)
           x = self.norm2(x, batch)
           x = nnx.relu(x)

           # Global pooling and classification
           x = global_mean_pool(x, batch)
           x = self.dropout(x)
           return self.classifier(x)

Module Organization
-------------------

**Convolution Layers** (``jraphx.nn.conv``):
   - ``MessagePassing``: Base class for custom layers
   - ``GCNConv``: Graph Convolutional Network
   - ``GATConv``: Graph Attention Network
   - ``GATv2Conv``: Improved GAT with dynamic attention
   - ``SAGEConv``: GraphSAGE with multiple aggregations
   - ``GINConv``: Graph Isomorphism Network
   - ``EdgeConv``: Edge convolution for point clouds
   - ``DynamicEdgeConv``: Dynamic edge convolution (requires pre-computed k-NN)
   - ``TransformerConv``: Graph Transformer with multi-head attention

**Pre-built Models** (``jraphx.nn.models``):
   - ``GCN``: Multi-layer GCN architecture
   - ``GAT``: Multi-layer GAT architecture
   - ``GraphSAGE``: Multi-layer GraphSAGE architecture
   - ``GIN``: Multi-layer GIN architecture
   - ``MLP``: Multi-layer perceptron
   - ``JumpingKnowledge``: Layer aggregation module
   - ``BasicGNN``: Abstract base class for GNN models

**Normalization** (``jraphx.nn.norm``):
   - ``BatchNorm``: Batch normalization with running statistics
   - ``LayerNorm``: Layer normalization (node-wise or graph-wise)
   - ``GraphNorm``: Graph-specific normalization

**Pooling** (``jraphx.nn.pool``):
   - ``global_add_pool``: Sum aggregation
   - ``global_mean_pool``: Mean aggregation
   - ``global_max_pool``: Max aggregation
   - ``global_min_pool``: Min aggregation
   - ``TopKPooling``: Select top-k important nodes
   - ``SAGPooling``: Self-attention graph pooling
