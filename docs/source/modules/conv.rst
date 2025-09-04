jraphx.nn.conv
==============

.. currentmodule:: jraphx.nn.conv

This module contains graph convolution layers implementing various message passing algorithms.

Core Message Passing Framework
------------------------------

MessagePassing
~~~~~~~~~~~~~~

.. autoclass:: MessagePassing
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all graph neural network layers implementing the message passing paradigm.

   **Message Passing Steps:**

   1. **Message**: Compute messages from neighboring nodes
   2. **Aggregate**: Aggregate messages using sum, mean, max, or min
   3. **Update**: Update node representations based on aggregated messages

   **Creating Custom Layers:**

   .. code-block:: python

      from jraphx.nn.conv import MessagePassing
      import flax.nnx as nnx
      import jax.numpy as jnp

      class MyGNNLayer(MessagePassing):
          def __init__(self, in_features, out_features, rngs):
              super().__init__(aggr='mean')
              self.lin = nnx.Linear(in_features, out_features, rngs=rngs)

          def message(self, x_j, x_i=None, edge_attr=None):
              # x_j: Features of source nodes
              # x_i: Features of target nodes (optional)
              # edge_attr: Edge features (optional)
              return x_j

          def update(self, aggr_out, x):
              # aggr_out: Aggregated messages
              # x: Original node features
              return self.lin(jnp.concatenate([x, aggr_out], axis=-1))

Graph Convolution Layers
------------------------

GCNConv
~~~~~~~

.. autoclass:: GCNConv
   :members:
   :undoc-members:
   :show-inheritance:

   Graph Convolutional Network layer from `Kipf & Welling (2017) <https://arxiv.org/abs/1609.02907>`_.

   **Mathematical Formulation:**

   .. math::

      X' = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W)

   where :math:`\tilde{A} = A + I` is the adjacency matrix with self-loops and :math:`\tilde{D}` is the degree matrix.

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import GCNConv
      import flax.nnx as nnx

      conv = GCNConv(
          in_features=16,
          out_features=32,
          add_self_loops=True,
          normalize=True,
          bias=True,
          rngs=nnx.Rngs(0)
      )

      out = conv(x, edge_index)

GATConv
~~~~~~~

.. autoclass:: GATConv
   :members:
   :undoc-members:
   :show-inheritance:

   Graph Attention Network layer from `Veličković et al. (2018) <https://arxiv.org/abs/1710.10903>`_.

   **Attention Mechanism:**

   .. math::

      \alpha_{ij} = \text{softmax}_j(e_{ij})

      e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j])

   **Multi-head Attention:**

   - Multiple attention heads compute independent attention weights
   - Outputs can be concatenated or averaged

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import GATConv
      import flax.nnx as nnx

      conv = GATConv(
          in_features=16,
          out_features=32,
          heads=8,
          concat=True,  # Concatenate head outputs
          dropout=0.6,
          add_self_loops=True,
          rngs=nnx.Rngs(0)
      )

      out = conv(x, edge_index)
      # Output shape: [num_nodes, heads * out_features] if concat=True

GATv2Conv
~~~~~~~~~

.. autoclass:: GATv2Conv
   :members:
   :undoc-members:
   :show-inheritance:

   Improved Graph Attention Network layer from `Brody et al. (2022) <https://arxiv.org/abs/2105.14491>`_.

   **Key Improvements over GAT:**

   - Dynamic attention: Attention weights depend on both query and key node features
   - More expressive: Can learn more complex attention patterns
   - Better performance: Often outperforms original GAT

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import GATv2Conv
      import flax.nnx as nnx

      conv = GATv2Conv(
          in_features=16,
          out_features=32,
          heads=8,
          concat=True,
          dropout=0.6,
          edge_dim=8,  # Optional edge features
          rngs=nnx.Rngs(0)
      )

      out = conv(x, edge_index, edge_attr=edge_attr)

SAGEConv
~~~~~~~~

.. autoclass:: SAGEConv
   :members:
   :undoc-members:
   :show-inheritance:

   GraphSAGE layer from `Hamilton et al. (2017) <https://arxiv.org/abs/1706.02216>`_.

   **Aggregation Options:**

   - **mean**: Average neighbor features
   - **max**: Element-wise maximum
   - **lstm**: LSTM aggregation over neighbors

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import SAGEConv
      import flax.nnx as nnx

      # Mean aggregation (most common)
      conv = SAGEConv(
          in_features=16,
          out_features=32,
          aggr='mean',
          normalize=True,
          rngs=nnx.Rngs(0)
      )

      # LSTM aggregation
      conv_lstm = SAGEConv(
          in_features=16,
          out_features=32,
          aggr='lstm',
          rngs=nnx.Rngs(0)
      )

      out = conv(x, edge_index)

GINConv
~~~~~~~

.. autoclass:: GINConv
   :members:
   :undoc-members:
   :show-inheritance:

   Graph Isomorphism Network layer from `Xu et al. (2019) <https://arxiv.org/abs/1810.00826>`_.

   **Key Features:**

   - Most expressive GNN under the WL-test framework
   - Uses MLPs for transformation
   - Learnable or fixed epsilon parameter

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import GINConv
      from jraphx.nn.models import MLP
      import flax.nnx as nnx

      # Create MLP for GIN
      mlp = MLP(
          channel_list=[16, 32, 32],
          norm="batch_norm",
          act="relu",
          rngs=nnx.Rngs(0)
      )

      conv = GINConv(
          nn=mlp,
          eps=0.0,
          train_eps=True  # Learn epsilon
      )

      out = conv(x, edge_index)

EdgeConv
~~~~~~~~

.. autoclass:: EdgeConv
   :members:
   :undoc-members:
   :show-inheritance:

   Dynamic edge convolution from `Wang et al. (2019) <https://arxiv.org/abs/1801.07829>`_.

   **Dynamic Graph Construction:**

   - Can dynamically compute k-nearest neighbors
   - Suitable for point cloud processing
   - Edge features computed from node pairs

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import EdgeConv
      from jraphx.nn.models import MLP
      import flax.nnx as nnx

      # MLP processes edge features [x_i || x_j - x_i]
      mlp = MLP(
          channel_list=[32, 64, 64],
          rngs=nnx.Rngs(0)
      )

      conv = EdgeConv(nn=mlp, aggr='max')
      out = conv(x, edge_index)

DynamicEdgeConv
~~~~~~~~~~~~~~~

.. autoclass:: DynamicEdgeConv
   :members:
   :undoc-members:
   :show-inheritance:

   Dynamic edge convolution from `Wang et al. (2019) <https://arxiv.org/abs/1801.07829>`_.

   **JraphX vs PyTorch Geometric:**

   - **PyG**: Automatically computes k-NN using ``torch_cluster.knn()``
   - **JraphX**: Requires pre-computed k-NN indices (simplified version)

   **Limitations:**

   - No automatic k-NN computation from node features
   - Requires external k-NN libraries (e.g., sklearn, faiss)
   - k-NN indices must be provided as input

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import DynamicEdgeConv
      from jraphx.nn.models import MLP
      import jax.numpy as jnp
      import flax.nnx as nnx

      # Create MLP for edge processing [x_i || x_j - x_i]
      mlp = MLP(
          channel_list=[6, 64, 128],  # Input: 2*3=6 for 3D points
          rngs=nnx.Rngs(0)
      )

      conv = DynamicEdgeConv(nn=mlp, k=6, aggr='max')

      # Pre-compute k-NN indices (6 nearest neighbors)
      # In practice, use sklearn.neighbors.NearestNeighbors or similar
      knn_indices = compute_knn_indices(x, k=6)

      out = conv(x, knn_indices=knn_indices)

TransformerConv
~~~~~~~~~~~~~~~

.. autoclass:: TransformerConv
   :members:
   :undoc-members:
   :show-inheritance:

   Graph Transformer layer from `Shi et al. (2021) <https://arxiv.org/abs/2009.03509>`_.

   **Multi-head Attention:**

   - Efficient QKV projection using single linear layer
   - Scaled dot-product attention
   - Optional edge feature incorporation
   - Beta gating mechanism for skip connections

   **Example:**

   .. code-block:: python

      from jraphx.nn.conv import TransformerConv
      import flax.nnx as nnx

      conv = TransformerConv(
          in_features=16,
          out_features=32,
          heads=8,
          concat=True,
          dropout_rate=0.1,
          edge_dim=8,  # Optional edge features
          beta=True,  # Gating mechanism
          root_weight=True,  # Skip connection
          rngs=nnx.Rngs(0)
      )

      out = conv(x, edge_index, edge_attr=edge_attr)

Layer Selection Guide
---------------------

Choosing the Right Layer
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Layer Comparison
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Layer
     - Complexity
     - Expressiveness
     - Memory Usage
     - Best For
   * - GCNConv
     - Low
     - Medium
     - Low
     - Citation networks
   * - GATConv
     - Medium
     - High
     - Medium
     - Heterophilic graphs
   * - GATv2Conv
     - Medium
     - Higher
     - Medium
     - Complex attention patterns
   * - SAGEConv
     - Low-Medium
     - Medium
     - Low-Medium
     - Large-scale graphs
   * - GINConv
     - Medium
     - Highest
     - Medium
     - Graph classification
   * - EdgeConv
     - High
     - High
     - High
     - Point clouds
   * - DynamicEdgeConv
     - High
     - High
     - High
     - Point clouds (k-NN)
   * - TransformerConv
     - High
     - Highest
     - High
     - Complex relationships

Performance Tips
~~~~~~~~~~~~~~~~

**Batch Processing:**

.. code-block:: python

   from jraphx.data import Batch

   # Batch multiple graphs for efficiency
   batch = Batch.from_data_list([graph1, graph2, graph3])
   out = conv(batch.x, batch.edge_index)

**JIT Compilation:**

.. code-block:: python

   import jax

   # JIT compile the forward pass
   @jax.jit
   def forward(x, edge_index):
       return conv(x, edge_index)

   out = forward(x, edge_index)

**Memory Efficiency:**

- Use ``concat=False`` in attention layers to reduce memory
- Consider ``aggr='mean'`` over ``aggr='lstm'`` for large graphs
- Use sparse operations when available

Edge Features
-------------

Many layers support edge features:

.. code-block:: python

   # GATv2 with edge features
   conv = GATv2Conv(16, 32, heads=8, edge_dim=4)
   out = conv(x, edge_index, edge_attr=edge_features)

   # TransformerConv with edge features
   conv = TransformerConv(16, 32, heads=8, edge_dim=4)
   out = conv(x, edge_index, edge_attr=edge_features)

Advanced Usage
--------------

Custom Aggregation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomConv(MessagePassing):
       def __init__(self, in_features, out_features):
           # Custom aggregation function
           super().__init__(aggr='add')

       def aggregate(self, inputs, index, dim_size=None):
           # Override for custom aggregation
           return scatter_mean(inputs, index, dim=0, dim_size=dim_size)

Heterogeneous Graphs
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different edge types
   edge_index_1 = ...  # Type 1 edges
   edge_index_2 = ...  # Type 2 edges

   # Use different convolutions
   conv1 = GCNConv(16, 32)
   conv2 = SAGEConv(16, 32)

   out1 = conv1(x, edge_index_1)
   out2 = conv2(x, edge_index_2)
   out = out1 + out2  # Combine
