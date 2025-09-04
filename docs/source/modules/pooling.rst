jraphx.nn.pool
==============

.. currentmodule:: jraphx.nn.pool

This module contains pooling operations for graph neural networks, including global pooling and hierarchical pooling methods.

Global Pooling Operations
-------------------------

Global pooling aggregates node features across entire graphs, producing graph-level representations.

global_add_pool
~~~~~~~~~~~~~~~

.. autofunction:: global_add_pool

   Sum all node features in each graph.

   **Example:**

   .. code-block:: python

      from jraphx.nn.pool import global_add_pool

      # For batched graphs
      graph_features = global_add_pool(x, batch)
      # Output shape: [num_graphs, features]

global_mean_pool
~~~~~~~~~~~~~~~~

.. autofunction:: global_mean_pool

   Average all node features in each graph.

   **Example:**

   .. code-block:: python

      from jraphx.nn.pool import global_mean_pool

      # Most common for graph classification
      graph_features = global_mean_pool(x, batch)

global_max_pool
~~~~~~~~~~~~~~~

.. autofunction:: global_max_pool

   Take element-wise maximum across all nodes in each graph.

   **Example:**

   .. code-block:: python

      from jraphx.nn.pool import global_max_pool

      # Good for capturing dominant features
      graph_features = global_max_pool(x, batch)

global_min_pool
~~~~~~~~~~~~~~~

.. autofunction:: global_min_pool

   Take element-wise minimum across all nodes in each graph.

   **Example:**

   .. code-block:: python

      from jraphx.nn.pool import global_min_pool

      # Less common but useful for specific tasks
      graph_features = global_min_pool(x, batch)

Hierarchical Pooling Layers
---------------------------

Hierarchical pooling layers select important nodes and create coarsened graph representations.

TopKPooling
~~~~~~~~~~~

.. autoclass:: TopKPooling
   :members:
   :undoc-members:
   :show-inheritance:

   Top-K pooling layer that selects the most important nodes based on learnable scores.

   **Algorithm:**

   1. Compute node scores using a learnable projection
   2. Select top-k nodes based on scores
   3. Update node features by multiplying with scores
   4. Filter edges to maintain graph connectivity

   **Example:**

   .. code-block:: python

      from jraphx.nn.pool import TopKPooling
      import flax.nnx as nnx

      # Select top 50% of nodes
      pool = TopKPooling(
          in_features=64,
          ratio=0.5,
          min_score=None,  # Optional minimum score threshold
          multiplier=1.0,  # Score multiplier
          rngs=nnx.Rngs(0)
      )

      # Apply pooling
      x_pool, edge_index_pool, edge_attr_pool, batch_pool, perm = pool(
          x, edge_index, edge_attr=edge_attr, batch=batch
      )

      # perm contains indices of selected nodes

   **Parameters Explained:**

   - **ratio**: If < 1, fraction of nodes to keep; if >= 1, exact number of nodes
   - **min_score**: Minimum score threshold (nodes below are filtered)
   - **multiplier**: Multiply scores before selection (affects gradients)

SAGPooling
~~~~~~~~~~

.. autoclass:: SAGPooling
   :members:
   :undoc-members:
   :show-inheritance:

   Self-attention graph pooling using GNN layers to compute importance scores.

   **Key Features:**

   - Uses GNN layers (GCN, GAT, or SAGE) to compute scores
   - More expressive than simple projection
   - Can capture graph structure in scoring

   **Example:**

   .. code-block:: python

      from jraphx.nn.pool import SAGPooling
      import flax.nnx as nnx

      # SAGPooling with GCN scoring
      pool = SAGPooling(
          in_features=64,
          ratio=0.5,
          gnn="gcn",  # Options: "gcn", "gat", "sage"
          min_score=None,
          multiplier=1.0,
          rngs=nnx.Rngs(0)
      )

      x_pool, edge_index_pool, edge_attr_pool, batch_pool, perm = pool(
          x, edge_index, edge_attr=edge_attr, batch=batch
      )

      # Using GAT for attention-based scoring
      pool_gat = SAGPooling(
          in_features=64,
          ratio=0.3,
          gnn="gat",
          rngs=nnx.Rngs(0)
      )

Pooling Strategies
------------------

Graph Classification Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.nn.conv import GCNConv
   from jraphx.nn.pool import TopKPooling, global_mean_pool
   import flax.nnx as nnx

   class GraphClassifier(nnx.Module):
       def __init__(self, in_features, num_classes, rngs):
           self.conv1 = GCNConv(in_features, 64, rngs=rngs)
           self.pool1 = TopKPooling(64, ratio=0.8, rngs=rngs)
           self.conv2 = GCNConv(64, 64, rngs=rngs)
           self.pool2 = TopKPooling(64, ratio=0.8, rngs=rngs)
           self.conv3 = GCNConv(64, 64, rngs=rngs)
           self.classifier = nnx.Linear(64, num_classes, rngs=rngs)
           self.dropout = nnx.Dropout(0.5, rngs=rngs)

       def __call__(self, x, edge_index, batch):
           # First GNN layer
           x = nnx.relu(self.conv1(x, edge_index))

           # First pooling
           x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)

           # Second GNN layer
           x = nnx.relu(self.conv2(x, edge_index))

           # Second pooling
           x, edge_index, _, batch, _ = self.pool2(x, edge_index, batch=batch)

           # Third GNN layer
           x = nnx.relu(self.conv3(x, edge_index))

           # Global pooling
           x = global_mean_pool(x, batch)

           # Classification
           x = self.dropout(x)
           return self.classifier(x)

Multi-Scale Pooling
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiScaleGNN(nnx.Module):
       def __init__(self, in_features, out_features, rngs):
           self.conv1 = GCNConv(in_features, 64, rngs=rngs)
           self.conv2 = GCNConv(64, 64, rngs=rngs)
           self.conv3 = GCNConv(64, 64, rngs=rngs)
           self.pool = TopKPooling(64, ratio=0.5, rngs=rngs)
           self.lin = nnx.Linear(192, out_features, rngs=rngs)

       def __call__(self, x, edge_index, batch):
           # Compute representations at multiple scales
           x1 = nnx.relu(self.conv1(x, edge_index))
           g1 = global_mean_pool(x1, batch)

           # Pool and compute second scale
           x2, edge_index2, _, batch2, _ = self.pool(x1, edge_index, batch=batch)
           x2 = nnx.relu(self.conv2(x2, edge_index2))
           g2 = global_mean_pool(x2, batch2)

           # Pool again and compute third scale
           x3, edge_index3, _, batch3, _ = self.pool(x2, edge_index2, batch=batch2)
           x3 = nnx.relu(self.conv3(x3, edge_index3))
           g3 = global_mean_pool(x3, batch3)

           # Concatenate multi-scale features
           out = jnp.concatenate([g1, g2, g3], axis=-1)
           return self.lin(out)

Pooling Selection Guide
-----------------------

Global Pooling
~~~~~~~~~~~~~~

.. list-table:: Global Pooling Methods
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Properties
     - Advantages
     - Use Cases
   * - add_pool
     - Sum aggregation
     - Preserves magnitude
     - Counting tasks
   * - mean_pool
     - Average aggregation
     - Size invariant
     - Most common, stable
   * - max_pool
     - Maximum values
     - Captures peaks
     - Dominant features
   * - min_pool
     - Minimum values
     - Captures valleys
     - Outlier detection

Hierarchical Pooling
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Hierarchical Pooling Methods
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Scoring Method
     - Complexity
     - Best For
   * - TopKPooling
     - Linear projection
     - Low
     - Fast coarsening
   * - SAGPooling (GCN)
     - GCN layer
     - Medium
     - Structure-aware
   * - SAGPooling (GAT)
     - GAT layer
     - High
     - Attention-based
   * - SAGPooling (SAGE)
     - SAGE layer
     - Medium
     - Neighbor aggregation

Performance Considerations
--------------------------

Memory Efficiency
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Aggressive pooling for memory efficiency
   pool = TopKPooling(64, ratio=0.1, rngs=nnx.Rngs(42))  # Keep only 10% of nodes

   # Gradual pooling for better gradients
   pool1 = TopKPooling(64, ratio=0.8, rngs=nnx.Rngs(42))  # First layer: 80%
   pool2 = TopKPooling(64, ratio=0.6, rngs=nnx.Rngs(42))  # Second layer: 60%

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.data import Batch

   # Efficient batched pooling
   batch_data = Batch.from_data_list(graphs)

   # Pool entire batch at once
   pooled_x, pooled_edge_index, _, pooled_batch, perm = pool(
       batch_data.x,
       batch_data.edge_index,
       batch=batch_data.batch
   )

JIT Compilation
~~~~~~~~~~~~~~~

.. code-block:: python

   import jax

   @jax.jit
   def pool_and_classify(x, edge_index, batch):
       # Pooling operations are JIT-compatible
       x_pool, edge_pool, _, batch_pool, _ = pool(x, edge_index, batch=batch)
       graph_features = global_mean_pool(x_pool, batch_pool)
       return classifier(graph_features)

Common Patterns
---------------

Differentiable Pooling
~~~~~~~~~~~~~~~~~~~~~~

All pooling operations maintain differentiability:

.. code-block:: python

   def loss_fn(params, x, edge_index, batch, y):
       # Pooling in computation graph
       x_pool, edge_pool, _, batch_pool, _ = pool(x, edge_index, batch=batch)
       graph_rep = global_mean_pool(x_pool, batch_pool)
       pred = classifier(graph_rep)
       return jnp.mean((pred - y) ** 2)

   # Gradients flow through pooling
   grads = jax.grad(loss_fn)(params, x, edge_index, batch, y)

Attention Visualization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get pooling scores for visualization
   pool = TopKPooling(64, ratio=0.5, rngs=nnx.Rngs(42))
   x_pool, _, _, _, perm = pool(x, edge_index)

   # perm contains indices of top nodes
   # Can visualize which nodes were selected
   selected_nodes = jnp.zeros(num_nodes)
   selected_nodes = selected_nodes.at[perm].set(1.0)
