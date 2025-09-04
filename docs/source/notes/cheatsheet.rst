:orphan:

JraphX GNN Cheatsheet
=====================

This cheatsheet provides an overview of all available Graph Neural Network layers in **JraphX** and their supported features.

**Legend:**

* :obj:`edge_weight`: If checked (✓), supports message passing with one-dimensional edge weight information, *e.g.*, :obj:`GCNConv(...)(x, edge_index, edge_weight)`.

* :obj:`edge_attr`: If checked (✓), supports message passing with multi-dimensional edge feature information, *e.g.*, :obj:`GATConv(...)(x, edge_index, edge_attr)`.

* **bipartite**: If checked (✓), supports message passing in bipartite graphs with potentially different feature dimensionalities for source and destination nodes.

* **JIT-ready**: If checked (✓), the layer is fully compatible with :obj:`@jax.jit` compilation for optimal performance.

* **vmap-ready**: If checked (✓), the layer can be efficiently vectorized over multiple graphs using :obj:`jax.vmap`.

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

JAX-Specific Optimizations
---------------------------

**JraphX** layers are designed to take full advantage of JAX's capabilities:

* **JIT Compilation**: All layers support :obj:`@jax.jit` for optimal performance
* **Vectorization**: Use :obj:`jax.vmap` to process multiple graphs in parallel
* **Functional Programming**: Pure functions with no side effects
* **Automatic Differentiation**: Full support for :obj:`jax.grad` and optimization libraries like Optax
* **XLA Backend**: Automatically optimized for your hardware (CPU/GPU/TPU)
