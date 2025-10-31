Data Structures & Utilities Cheatsheet
=======================================

This cheatsheet covers **JraphX**'s core data structures and utility functions for working with graphs.

.. note::

    **JraphX** is focused on providing GNN layers and utilities for JAX, not datasets. For datasets, you'll typically load data from external sources (files, other libraries) and convert them to JraphX format.

Core Data Structures
--------------------

.. list-table::
    :widths: 30 35 35
    :header-rows: 1

    * - Class
      - Purpose
      - Key Methods
    * - :class:`~jraphx.data.Data`
      - Single graph representation
      - :obj:`num_nodes`, :obj:`num_edges`, :obj:`keys()`, :obj:`__getitem__`
    * - :class:`~jraphx.data.Batch`
      - Multiple graphs in a batch
      - :obj:`from_data_list()`, :obj:`num_graphs`, :obj:`batch`

Data Attributes
---------------

.. list-table::
    :widths: 20 20 30 30
    :header-rows: 1

    * - Attribute
      - Shape
      - Description
      - Required
    * - :obj:`x`
      - :obj:`[num_nodes, num_features]`
      - Node feature matrix
      - Optional
    * - :obj:`edge_index`
      - :obj:`[2, num_edges]`
      - Edge connectivity in COO format
      - Optional
    * - :obj:`edge_attr`
      - :obj:`[num_edges, num_edge_features]`
      - Edge feature matrix
      - Optional
    * - :obj:`y`
      - :obj:`[num_nodes, *]` or :obj:`[num_graphs, *]`
      - Labels/targets
      - Optional
    * - :obj:`pos`
      - :obj:`[num_nodes, num_dimensions]`
      - Node positions (3D point clouds)
      - Optional
    * - :obj:`batch`
      - :obj:`[num_nodes]`
      - Batch assignment vector
      - Auto-generated

Graph Utility Functions
-----------------------

.. list-table::
    :widths: 35 40 25
    :header-rows: 1

    * - Function
      - Purpose
      - JIT-ready
    * - :func:`~jraphx.utils.add_self_loops`
      - Add self-loop edges to graph
      - ✓
    * - :func:`~jraphx.utils.remove_self_loops`
      - Remove self-loop edges from graph
      - ✓
    * - :func:`~jraphx.utils.degree`
      - Compute node degrees
      - ✓
    * - :func:`~jraphx.utils.in_degree`
      - Compute in-degrees (directed graphs)
      - ✓
    * - :func:`~jraphx.utils.out_degree`
      - Compute out-degrees (directed graphs)
      - ✓
    * - :func:`~jraphx.utils.coalesce`
      - Remove duplicate edges
      - ✓
    * - :func:`~jraphx.utils.to_undirected`
      - Convert directed to undirected graph
      - ✓
    * - :func:`~jraphx.utils.to_dense_adj`
      - Convert edge_index to dense adjacency
      - ✓
    * - :func:`~jraphx.utils.to_edge_index`
      - Convert dense adjacency to edge_index
      - ✓

Scatter Operations
------------------

.. list-table::
    :widths: 35 40 25
    :header-rows: 1

    * - Function
      - Purpose
      - JIT-ready
    * - :func:`~jraphx.utils.scatter_add`
      - Scatter-add operation for aggregation
      - ✓
    * - :func:`~jraphx.utils.scatter_mean`
      - Scatter-mean operation for aggregation
      - ✓
    * - :func:`~jraphx.utils.scatter_max`
      - Scatter-max operation for aggregation
      - ✓
    * - :func:`~jraphx.utils.scatter_min`
      - Scatter-min operation for aggregation
      - ✓
    * - :func:`~jraphx.utils.scatter_std`
      - Scatter-std operation for aggregation
      - ✓
    * - :func:`~jraphx.utils.scatter_logsumexp`
      - Scatter-logsumexp for numerical stability
      - ✓
    * - :func:`~jraphx.utils.scatter_softmax`
      - Scatter-softmax for attention mechanisms
      - ✓
    * - :func:`~jraphx.utils.scatter_log_softmax`
      - Scatter-log-softmax for attention
      - ✓

Quick Examples
--------------

**Creating a simple graph:**

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data

    # Create node features and edges
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])  # Triangle graph

    data = Data(x=x, edge_index=edge_index)
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

**Batching multiple graphs:**

.. code-block:: python

    from jraphx.data import Batch

    # Create list of graphs
    graphs = [
        Data(x=jnp.ones((3, 2)), edge_index=jnp.array([[0, 1], [1, 2]])),
        Data(x=jnp.ones((4, 2)), edge_index=jnp.array([[0, 1], [2, 3]])),
    ]

    batch = Batch.from_data_list(graphs)
    print(f"Batch has {batch.num_graphs} graphs")

**Using utilities:**

.. code-block:: python

    from jraphx.utils import add_self_loops, degree

    # Add self-loops
    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=3)

    # Compute degrees
    degrees = degree(edge_index[1], num_nodes=3)
    print(f"Node degrees: {degrees}")

**JIT compilation:**

.. code-block:: python

    import jax

    @jax.jit
    def process_graph(data):
        from jraphx.utils import add_self_loops
        edge_index, _ = add_self_loops(data.edge_index, data.x.shape[0])
        return edge_index

    processed = process_graph(data)

Working with PyTorch Geometric Datasets
---------------------------------------

You can easily use PyTorch Geometric datasets with **JraphX** by converting the data format:

**Loading a PyG dataset:**

.. code-block:: python

    import torch
    from torch_geometric.datasets import Cora
    import jax.numpy as jnp
    from jraphx.data import Data

    def pyg_to_jraphx(pyg_data):
        """Convert PyG Data to JraphX Data."""
        return Data(
            x=jnp.array(pyg_data.x.numpy()),
            edge_index=jnp.array(pyg_data.edge_index.numpy()),
            y=jnp.array(pyg_data.y.numpy()) if pyg_data.y is not None else None,
            edge_attr=jnp.array(pyg_data.edge_attr.numpy()) if pyg_data.edge_attr is not None else None,
        )

    # Load Cora dataset
    dataset = Cora(root='/tmp/Cora')
    pyg_data = dataset[0]  # Single graph dataset

    # Convert to JraphX format
    jraphx_data = pyg_to_jraphx(pyg_data)
    print(f"Converted graph: {jraphx_data.num_nodes} nodes, {jraphx_data.num_edges} edges")

**Batch processing multiple PyG graphs:**

.. code-block:: python

    from torch_geometric.datasets import TUDataset
    from jraphx.data import Batch

    # Load graph classification dataset
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    # Convert first 10 graphs to JraphX format
    jraphx_graphs = []
    for i in range(10):
        pyg_graph = dataset[i]
        jraphx_graph = pyg_to_jraphx(pyg_graph)
        jraphx_graphs.append(jraphx_graph)

    # Create batch for JraphX processing
    batch = Batch.from_data_list(jraphx_graphs)
    print(f"Batch contains {batch.num_graphs} graphs")

**Training with a PyG dataset:**

.. code-block:: python

    import jax
    import optax
    from flax import nnx
    from jraphx.nn.models import GCN
    from jraphx.nn.pool import global_mean_pool

    # Setup model for graph classification
    model = GCN(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        num_layers=3,
        rngs=nnx.Rngs(42)
    )

    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    @jax.jit
    def train_step(model, optimizer, batch, targets):
        def loss_fn(model):
            # Node-level predictions
            node_predictions = model(batch)
            # Pool to graph-level
            graph_predictions = global_mean_pool(node_predictions, batch.batch)
            # Compute loss
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                graph_predictions, targets
            ))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Training loop
    for epoch in range(100):
        # Sample batch of graphs
        indices = jnp.arange(min(32, len(dataset)))  # Batch size 32
        batch_graphs = [pyg_to_jraphx(dataset[i]) for i in indices]
        batch = Batch.from_data_list(batch_graphs)
        targets = jnp.array([dataset[i].y.item() for i in indices])

        loss = train_step(model, optimizer, batch, targets)
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

**Note on Normalization:**

JraphX normalization layers (BatchNorm, LayerNorm) follow Flax NNX conventions with the ``use_running_average`` parameter:

.. code-block:: python

    from jraphx.nn.norm import BatchNorm

    # Create graph-aware batch normalization
    bn = BatchNorm(in_features=64, rngs=rngs)

    # Training mode: model.train() causes BatchNorm to use use_running_average=False
    model.train()  # Sets training state
    x_train = bn(x, batch=batch)  # Automatically computes batch statistics

    # Evaluation mode: model.eval() causes BatchNorm to use use_running_average=True
    model.eval()   # Sets evaluation state
    x_eval = bn(x, batch=batch)   # Automatically uses running statistics

    # Manual control (overrides model state):
    x_manual = bn(x, batch=batch, use_running_average=False)  # Force batch stats

**Common PyG datasets for JraphX:**

.. list-table::
    :widths: 30 20 20 30
    :header-rows: 1

    * - Dataset
      - Type
      - Size
      - Use Case
    * - **Cora**
      - Citation Network
      - 2,708 nodes
      - Node classification
    * - **ENZYMES**
      - Graph Classification
      - 600 graphs
      - Graph classification
    * - **Karate Club**
      - Social Network
      - 34 nodes
      - Community detection
    * - **QM7**
      - Molecular
      - 7,165 molecules
      - Graph regression
    * - **Reddit**
      - Social Network
      - 232K nodes
      - Large-scale node classification

**Memory-efficient dataset loading:**

.. code-block:: python

    def lazy_pyg_to_jraphx_converter(dataset):
        """Generator that converts PyG graphs to JraphX format lazily."""
        for i in range(len(dataset)):
            yield pyg_to_jraphx(dataset[i])

    # Use with large datasets to avoid memory issues
    large_dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')

    # Process in batches
    batch_size = 32
    for batch_start in range(0, len(large_dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(large_dataset))
        batch_graphs = [
            pyg_to_jraphx(large_dataset[i])
            for i in range(batch_start, batch_end)
        ]
        batch = Batch.from_data_list(batch_graphs)
        # Process batch...
        print(f"Processed batch {batch_start//batch_size + 1}")

This integration allows you to leverage the extensive PyG dataset collection while using **JraphX**'s JAX-optimized graph neural networks.

Data Augmentation with Flax NNX
-----------------------------------

Use the new Rngs shorthand methods for data augmentation and preprocessing:

.. code-block:: python

    from flax import nnx

    def augment_graph_data(data, rngs):
        """Augment graph data using new Rngs shorthand methods."""

        # Add random noise to node features (traditional approach)
        # noise = random.normal(rngs(), data.x.shape) * 0.1

        # Use shorthand methods instead (Flax NNX)
        noise = rngs.normal(data.x.shape) * 0.1
        x_noisy = data.x + noise

        # Add random edge perturbations
        num_perturb = 5
        new_edges = jnp.stack([
            rngs.params.randint((num_perturb,), 0, data.num_nodes),
            rngs.params.randint((num_perturb,), 0, data.num_nodes)
        ])

        # Combine original and new edges
        edge_index_augmented = jnp.concatenate([data.edge_index, new_edges], axis=1)

        return data.replace(x=x_noisy, edge_index=edge_index_augmented)

    # Usage with multiple key streams
    rngs = nnx.Rngs(0, params=1, dropout=2)  # Named key streams
    augmented_data = augment_graph_data(original_data, rngs)
