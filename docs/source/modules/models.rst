jraphx.nn.models
================

.. currentmodule:: jraphx.nn.models

This module contains pre-built GNN model architectures that can be used out-of-the-box for common graph learning tasks.

Pre-built GNN Models
--------------------

These models provide complete architectures with multiple layers, normalization, dropout, and optional features like JumpingKnowledge connections.

GCN
~~~

.. autoclass:: GCN
   :members:
   :undoc-members:
   :show-inheritance:

   Graph Convolutional Network model with configurable layers and normalization.

   **Example:**

   .. code-block:: python

      from jraphx.nn.models import GCN
      import flax.nnx as nnx

      model = GCN(
          in_features=16,
          hidden_features=64,
          num_layers=3,
          out_features=10,
          dropout_rate=0.5,
          norm="layer_norm",  # Options: "batch_norm", "layer_norm", "graph_norm", None
          jk=None,  # Options: "cat", "max", "lstm", None
          rngs=nnx.Rngs(0)
      )

      # Forward pass
      out = model(x, edge_index, batch=batch)

GAT
~~~

.. autoclass:: GAT
   :members:
   :undoc-members:
   :show-inheritance:

   Graph Attention Network model with multi-head attention and configurable architecture.

   **Example:**

   .. code-block:: python

      from jraphx.nn.models import GAT
      import flax.nnx as nnx

      model = GAT(
          in_features=16,
          hidden_features=64,
          num_layers=3,
          out_features=10,
          heads=8,
          v2=False,  # Use GATv2 if True
          dropout_rate=0.6,
          norm="layer_norm",
          jk="max",  # JumpingKnowledge aggregation
          rngs=nnx.Rngs(0)
      )

      out = model(x, edge_index, batch=batch)

GraphSAGE
~~~~~~~~~

.. autoclass:: GraphSAGE
   :members:
   :undoc-members:
   :show-inheritance:

   GraphSAGE model with multiple aggregation options.

   **Example:**

   .. code-block:: python

      from jraphx.nn.models import GraphSAGE
      import flax.nnx as nnx

      model = GraphSAGE(
          in_features=16,
          hidden_features=64,
          num_layers=3,
          out_features=10,
          aggr="mean",  # Options: "mean", "max", "lstm"
          dropout_rate=0.5,
          norm="batch_norm",
          jk="cat",  # Concatenate all layer outputs
          rngs=nnx.Rngs(0)
      )

      out = model(x, edge_index, batch=batch)

GIN
~~~

.. autoclass:: GIN
   :members:
   :undoc-members:
   :show-inheritance:

   Graph Isomorphism Network model with MLP transformations.

   **Example:**

   .. code-block:: python

      from jraphx.nn.models import GIN
      import flax.nnx as nnx

      model = GIN(
          in_features=16,
          hidden_features=64,
          num_layers=5,
          out_features=10,
          dropout_rate=0.5,
          norm="batch_norm",
          jk="cat",
          rngs=nnx.Rngs(0)
      )

      out = model(x, edge_index, batch=batch)

Base Classes
------------

BasicGNN
~~~~~~~~

.. autoclass:: BasicGNN
   :members:
   :undoc-members:
   :show-inheritance:

   Abstract base class for GNN models. Provides a common interface for building multi-layer GNNs with normalization, dropout, and JumpingKnowledge connections.

   **Subclassing Example:**

   .. code-block:: python

      from jraphx.nn.models import BasicGNN
      from jraphx.nn.conv import MessagePassing

      class MyCustomGNN(BasicGNN):
          def init_conv(self, in_features, out_features, rngs=None, **kwargs):
              # Return your custom message passing layer
              return MyCustomConv(in_features, out_features, rngs=rngs, **kwargs)

Utility Models
--------------

MLP
~~~

.. autoclass:: MLP
   :members:
   :undoc-members:
   :show-inheritance:

   Multi-layer perceptron with configurable layers, normalization, and dropout.

   **Example:**

   .. code-block:: python

      from jraphx.nn.models import MLP
      import flax.nnx as nnx

      # Using channel list
      mlp = MLP(
          channel_list=[16, 64, 64, 32, 10],
          norm="layer_norm",
          bias=True,
          dropout_rate=0.5,
          act="relu",
          rngs=nnx.Rngs(0)
      )

      # Or using in/hidden/out channels
      mlp = MLP(
          in_features=16,
          hidden_features=64,
          out_features=10,
          num_layers=3,
          norm="batch_norm",
          dropout_rate=0.5,
          rngs=nnx.Rngs(0)
      )

      out = mlp(x)

JumpingKnowledge
~~~~~~~~~~~~~~~~

.. autoclass:: JumpingKnowledge
   :members:
   :undoc-members:
   :show-inheritance:

   JumpingKnowledge layer for aggregating representations from different GNN layers.

   **Example:**

   .. code-block:: python

      from jraphx.nn.models import JumpingKnowledge
      import flax.nnx as nnx

      # Concatenation mode
      jk = JumpingKnowledge(mode="cat", channels=64, num_layers=3)

      # Max pooling mode
      jk = JumpingKnowledge(mode="max")

      # LSTM aggregation mode
      jk = JumpingKnowledge(
          mode="lstm",
          channels=64,
          num_layers=3,
          rngs=nnx.Rngs(0)
      )

      # Aggregate layer outputs
      layer_outputs = [layer1_out, layer2_out, layer3_out]
      final_out = jk(layer_outputs)

Model Selection Guide
---------------------

Choosing the Right Model
~~~~~~~~~~~~~~~~~~~~~~~~

**GCN**: Best for citation networks and semi-supervised learning tasks with homophilic graphs.

**GAT**: Excellent for graphs where edge importance varies. The attention mechanism learns which neighbors are most relevant.

**GraphSAGE**: Ideal for large-scale graphs and inductive learning scenarios where you need to generalize to unseen nodes.

**GIN**: Most expressive for distinguishing graph structures. Best for graph-level tasks like molecular property prediction.

Configuration Tips
~~~~~~~~~~~~~~~~~~

**Number of Layers**:
   - 2-3 layers for most node classification tasks
   - 4-5 layers for graph-level tasks
   - Use JumpingKnowledge for deeper networks

**Normalization**:
   - ``batch_norm``: Best for large batches and stable training
   - ``layer_norm``: Works well with smaller batches
   - ``graph_norm``: Specifically designed for graph data

**JumpingKnowledge**:
   - ``cat``: Preserves all information but increases dimensionality
   - ``max``: Good balance of expressiveness and efficiency
   - ``lstm``: Most flexible but requires more parameters

**Dropout**:
   - 0.5-0.6 for training stability
   - Higher rates (0.6-0.8) for GAT models
   - Lower rates (0.2-0.5) for deeper models

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Performance Characteristics
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Model
     - Speed
     - Memory
     - Expressiveness
     - Best For
   * - GCN
     - Fast
     - Low
     - Medium
     - Node classification
   * - GAT
     - Medium
     - Medium-High
     - High
     - Heterophilic graphs
   * - GraphSAGE
     - Fast
     - Low-Medium
     - Medium
     - Large-scale graphs
   * - GIN
     - Fast
     - Low
     - Highest
     - Graph classification
