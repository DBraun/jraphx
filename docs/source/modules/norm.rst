jraphx.nn.norm
==============

.. currentmodule:: jraphx.nn.norm

This module contains normalization layers specifically designed for graph neural networks.

Normalization Layers
--------------------

BatchNorm
~~~~~~~~~

.. autoclass:: BatchNorm
   :members:
   :undoc-members:
   :show-inheritance:

   Batch normalization layer with running statistics for graph data.

   **Key Features:**

   - Maintains running mean and variance for inference
   - Configurable momentum and epsilon parameters
   - Supports both training and evaluation modes
   - Compatible with graph batching

   **Example:**

   .. code-block:: python

      from jraphx.nn.norm import BatchNorm
      import flax.nnx as nnx

      # Create batch norm layer
      norm = BatchNorm(
          num_features=64,
          eps=1e-5,
          momentum=0.99,
          affine=True,
          track_running_stats=True,
          rngs=nnx.Rngs(0)
      )

      # Apply normalization
      x_normalized = norm(x)  # Training mode

      # Switch to eval mode
      norm.eval()
      x_eval = norm(x)  # Uses running statistics

LayerNorm
~~~~~~~~~

.. autoclass:: LayerNorm
   :members:
   :undoc-members:
   :show-inheritance:

   Layer normalization for graph neural networks with node-wise or graph-wise modes.

   **Normalization Modes:**

   - **node**: Normalize across feature dimensions for each node independently
   - **graph**: Normalize across all nodes and features in a graph

   **Example:**

   .. code-block:: python

      from jraphx.nn.norm import LayerNorm
      import flax.nnx as nnx

      # Node-wise normalization
      norm = LayerNorm(
          num_features=64,
          mode="node",
          eps=1e-5,
          elementwise_affine=True,
          rngs=nnx.Rngs(0)
      )

      x_normalized = norm(x)

      # Graph-wise normalization (requires batch index)
      norm_graph = LayerNorm(
          num_features=64,
          mode="graph",
          rngs=nnx.Rngs(0)
      )

      x_normalized = norm_graph(x, batch=batch)

GraphNorm
~~~~~~~~~

.. autoclass:: GraphNorm
   :members:
   :undoc-members:
   :show-inheritance:

   Graph normalization layer that normalizes node features across the graph structure.

   **Algorithm:**

   1. Compute mean and variance per graph
   2. Normalize features within each graph
   3. Apply learnable affine transformation

   **Example:**

   .. code-block:: python

      from jraphx.nn.norm import GraphNorm
      import flax.nnx as nnx

      norm = GraphNorm(
          num_features=64,
          eps=1e-5,
          rngs=nnx.Rngs(0)
      )

      # For batched graphs
      x_normalized = norm(x, batch=batch)

      # For single graph (batch=None)
      x_normalized = norm(x)

Normalization Selection Guide
-----------------------------

When to Use Each Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**BatchNorm**:
   - Best with large, consistent batch sizes
   - Effective for deep networks
   - Requires sufficient batch statistics
   - Good for training stability

**LayerNorm**:
   - Works well with variable batch sizes
   - Effective for attention-based models
   - Node mode: Best for heterogeneous graphs
   - Graph mode: Best for homogeneous graphs

**GraphNorm**:
   - Specifically designed for graph data
   - Handles varying graph sizes well
   - Good for graph-level tasks
   - Robust to batch size variations

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Normalization Characteristics
   :header-rows: 1
   :widths: 25 25 25 25

   * - Normalization
     - Batch Size Sensitivity
     - Graph Size Sensitivity
     - Best Use Case
   * - BatchNorm
     - High
     - Low
     - Large batch training
   * - LayerNorm (node)
     - None
     - None
     - Node-level tasks
   * - LayerNorm (graph)
     - None
     - Medium
     - Small graphs
   * - GraphNorm
     - Low
     - Low
     - Graph-level tasks

Implementation Details
----------------------

Running Statistics
~~~~~~~~~~~~~~~~~~

BatchNorm maintains running statistics for inference:

.. code-block:: python

   # During training
   norm.train()  # Use batch statistics
   output = norm(x)

   # During inference
   norm.eval()  # Use running statistics
   output = norm(x)

Affine Transformations
~~~~~~~~~~~~~~~~~~~~~~

All normalization layers support learnable affine parameters:

.. code-block:: python

   # With affine transformation (default)
   norm = LayerNorm(64, elementwise_affine=True)
   # Learns gamma and beta parameters

   # Without affine transformation
   norm = LayerNorm(64, elementwise_affine=False)
   # Pure normalization only

Numerical Stability
~~~~~~~~~~~~~~~~~~~

All layers use epsilon for numerical stability:

.. code-block:: python

   # Adjust epsilon for precision
   norm = GraphNorm(64, eps=1e-5)  # Default
   norm = GraphNorm(64, eps=1e-8)  # Higher precision

Integration with GNN Models
---------------------------

Using with Pre-built Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.nn.models import GCN

   # GCN with layer normalization
   model = GCN(
       in_features=16,
       hidden_features=64,
       num_layers=3,
       out_features=10,
       norm="layer_norm",  # or "batch_norm", "graph_norm"
       rngs=nnx.Rngs(0)
   )

Custom Model Integration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.nn.conv import GCNConv
   from jraphx.nn.norm import GraphNorm
   import flax.nnx as nnx

   class CustomGNN(nnx.Module):
       def __init__(self, in_features, out_features, rngs):
           self.conv = GCNConv(in_features, 64, rngs=rngs)
           self.norm = GraphNorm(64, rngs=rngs)
           self.linear = nnx.Linear(64, out_features, rngs=rngs)

       def __call__(self, x, edge_index, batch=None):
           x = self.conv(x, edge_index)
           x = self.norm(x, batch)
           x = nnx.relu(x)
           return self.linear(x)
