JraphX GNN Cheatsheet
=====================

This cheatsheet provides an overview of all available Graph Neural Network layers in **JraphX** and their supported features.

**Legend:**

* :obj:`edge_weight`: If checked (✓), supports message passing with one-dimensional edge weight information, *e.g.*, :obj:`GCNConv(...)(x, edge_index, edge_weight)`.

* :obj:`edge_attr`: If checked (✓), supports message passing with multi-dimensional edge feature information, *e.g.*, :obj:`GATConv(...)(x, edge_index, edge_attr)`.

* **bipartite**: If checked (✓), supports message passing in bipartite graphs with potentially different feature dimensionalities for source and destination nodes.

* **JIT-ready**: If checked (✓), the layer is fully compatible with :obj:`@jax.jit` compilation for optimal performance.

* **vmap-ready**: If checked (✓), the layer can be efficiently vectorized over multiple graphs using :obj:`nnx.vmap`.

Graph Neural Network Operators
------------------------------

.. list-table::
    :widths: 30 15 15 15 15 15
    :header-rows: 1

    * - Name
      - :obj:`edge_weight`
      - :obj:`edge_attr`
      - bipartite
      - JIT-ready
      - vmap-ready
    * - :class:`~jraphx.nn.conv.GCNConv` (`Paper <https://arxiv.org/abs/1609.02907>`__)
      - ✓
      -
      -
      - ✓
      - ✓
    * - :class:`~jraphx.nn.conv.GATConv` (`Paper <https://arxiv.org/abs/1710.10903>`__)
      -
      - ✓
      -
      - ✓
      - ✓
    * - :class:`~jraphx.nn.conv.GATv2Conv` (`Paper <https://arxiv.org/abs/2105.14491>`__)
      -
      - ✓
      -
      - ✓
      - ✓
    * - :class:`~jraphx.nn.conv.SAGEConv` (`Paper <https://arxiv.org/abs/1706.02216>`__)
      -
      -
      -
      - ✓
      - ✓
    * - :class:`~jraphx.nn.conv.GINConv` (`Paper <https://arxiv.org/abs/1810.00826>`__)
      -
      -
      -
      - ✓
      - ✓
    * - :class:`~jraphx.nn.conv.EdgeConv` (`Paper <https://arxiv.org/abs/1801.07829>`__)
      -
      -
      -
      - ✓
      - ✓
    * - :class:`~jraphx.nn.conv.TransformerConv` (`Paper <https://arxiv.org/abs/2012.09699>`__)
      -
      - ✓
      -
      - ✓
      - ✓

Pre-built Models
----------------

**JraphX** provides several pre-built GNN models that combine multiple layers:

.. list-table::
    :widths: 50 25 25
    :header-rows: 1

    * - Name
      - JIT-ready
      - vmap-ready
    * - :class:`~jraphx.nn.models.GCN`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.models.GAT`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.models.GraphSAGE`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.models.GIN`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.models.MLP`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.models.JumpingKnowledge`
      - ✓
      - ✓

Normalization Layers
--------------------

.. list-table::
    :widths: 50 25 25
    :header-rows: 1

    * - Name
      - JIT-ready
      - vmap-ready
    * - :class:`~jraphx.nn.norm.BatchNorm`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.norm.LayerNorm`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.norm.GraphNorm`
      - ✓
      - ✓

Pooling Operations
------------------

.. list-table::
    :widths: 50 25 25
    :header-rows: 1

    * - Name
      - JIT-ready
      - vmap-ready
    * - :func:`~jraphx.nn.pool.global_add_pool`
      - ✓
      - ✓
    * - :func:`~jraphx.nn.pool.global_mean_pool`
      - ✓
      - ✓
    * - :func:`~jraphx.nn.pool.global_max_pool`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.pool.TopKPooling`
      - ✓
      - ✓
    * - :class:`~jraphx.nn.pool.SAGPooling`
      - ✓
      - ✓

Quick Usage Examples
--------------------

**Basic layer usage:**

.. code-block:: python

    import jax.numpy as jnp
    from flax import nnx
    from jraphx.nn.conv import GCNConv, GATConv, EdgeConv
    from jraphx.data import Data
    from jraphx.nn.models import MLP

    # Create graph data
    x = jnp.ones((10, 16))
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])
    data = Data(x=x, edge_index=edge_index)

    # GCN layer (supports edge weights)
    gcn = GCNConv(16, 32, rngs=nnx.Rngs(42))
    gcn_out = gcn(data.x, data.edge_index)

    # GAT layer (supports edge attributes)
    gat = GATConv(16, 32, heads=4, rngs=nnx.Rngs(42))
    gat_out = gat(data.x, data.edge_index)

    # EdgeConv layer (requires neural network module)
    edge_mlp = MLP([32, 32, 32], rngs=nnx.Rngs(42))  # 2*16 -> 32 -> 32
    edge_conv = EdgeConv(edge_mlp, aggr='max')
    edge_out = edge_conv(data.x, data.edge_index)

**Pre-built model usage:**

.. code-block:: python

    from jraphx.nn.models import GCN

    # Create multi-layer GCN
    model = GCN(
        in_features=16,
        hidden_features=64,
        out_features=7,
        num_layers=3,
        dropout=0.1,
        rngs=nnx.Rngs(42)
    )

    # Forward pass
    predictions = model(data.x, data.edge_index)

**Pooling for graph-level tasks:**

.. code-block:: python

    from jraphx.nn.pool import global_mean_pool
    from jraphx.data import Batch

    # Create batch of graphs
    graphs = [data, data, data]  # 3 identical graphs for demo
    batch = Batch.from_data_list(graphs)

    # Get node-level features
    node_features = model(batch.x, batch.edge_index)

    # Pool to graph-level representations
    graph_features = global_mean_pool(node_features, batch.batch)
    print(f"Graph features: {graph_features.shape}")  # [3, feature_dim]

JAX-Specific Optimizations
---------------------------

**JraphX** layers are designed to take full advantage of JAX's capabilities:

* **JIT Compilation**: All layers support :obj:`@jax.jit` for optimal performance
* **Vectorization**: Use :obj:`nnx.vmap` to process multiple graphs in parallel
* **Automatic Differentiation**: Full support for :obj:`jax.grad` and optimization libraries like Optax
* **XLA Backend**: Automatically optimized for your hardware (CPU/GPU/TPU)

**Performance example:**

.. code-block:: python

    import jax

    # JIT compile for speed
    @jax.jit
    def fast_gnn_inference(model, x, edge_index):
        return model(x, edge_index)

    # Vectorize over multiple graphs (fixed-size)
    @nnx.vmap
    def batch_gnn_inference(x_batch, edge_index_batch):
        return model(x_batch, edge_index_batch)

    # Use with optimization libraries
    import optax
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    @jax.jit
    def train_step(model, optimizer, data, targets):
        def loss_fn(model):
            preds = model(data.x, data.edge_index)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

Random Number Generation (Flax NNX)
--------------------------------------

Use modern **Flax NNX** Rngs shorthand methods for cleaner code:

.. code-block:: python

    # Create Rngs with named key streams
    rngs = nnx.Rngs(0, params=1, dropout=2)

    # Old JAX approach:
    # noise = random.normal(random.key(42), (10, 16))

    # New Flax shorthand (much cleaner!):
    noise = rngs.normal((10, 16))                    # Default key
    features = rngs.params.uniform((10, 16))         # Params key
    dropout_mask = rngs.dropout.bernoulli(0.5, (10,))  # Dropout key

For more details, see the `Flax randomness guide <https://flax.readthedocs.io/en/latest/guides/randomness.html#jax-random-shorthand-methods>`__.

Missing Features
----------------

For a complete list of PyTorch Geometric features not yet implemented in **JraphX**, see :doc:`../missing_features`.
