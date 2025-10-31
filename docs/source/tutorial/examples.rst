Practical Examples
==================

This section provides complete, self-contained examples to help you get started with JraphX. Each example demonstrates different aspects of graph neural networks and includes data generation, model definition, training, and evaluation.

.. contents:: Contents
    :local:

For more advanced examples and real-world use cases, check out the ``examples/`` directory in the JraphX repository, which contains working scripts for:

- **Node classification on Cora dataset** (``cora_planetoid.py``)
- **Graph attention networks** (``gat_example.py``)
- **Karate club clustering** (``karate_club.py``)
- **Pre-built model usage** (``pre_built_models.py``)
- **Advanced JAX transformations** (``nnx_transforms.py``)
- **And many more!**

Simple Graph Construction
-------------------------

Creating and manipulating basic graphs:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data

    # Create a simple triangle graph
    x = jnp.array([[1.0, 0.0],   # Node 0
                   [0.0, 1.0],   # Node 1
                   [1.0, 1.0]])  # Node 2

    edge_index = jnp.array([[0, 1, 2, 0],  # Source nodes
                            [1, 2, 0, 2]]) # Target nodes

    data = Data(x=x, edge_index=edge_index)

    print(f"Graph with {data.num_nodes} nodes and {data.num_edges} edges")

Basic GCN Model
---------------

A simple two-layer GCN for node classification:

.. code-block:: python

    import flax.nnx as nnx
    from jraphx.nn.conv import GCNConv

    class BasicGCN(nnx.Module):
        def __init__(self, in_features, hidden_dim, num_classes, rngs):
            self.conv1 = GCNConv(in_features, hidden_dim, rngs=rngs)
            self.conv2 = GCNConv(hidden_dim, num_classes, rngs=rngs)
            self.dropout = nnx.Dropout(0.5, rngs=rngs)

        def __call__(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = nnx.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return nnx.log_softmax(x, axis=-1)

    # Initialize model
    model = BasicGCN(
        in_features=data.num_node_features,
        hidden_dim=16,
        num_classes=3,
        rngs=nnx.Rngs(0)
    )

    # Forward pass
    output = model(data.x, data.edge_index)
    print(f"Output shape: {output.shape}")

Node Classification Example
----------------------------

Complete example for node classification:

.. code-block:: python

    import jax
    import optax
    from jraphx.data import Data

    # Create synthetic data
    def create_synthetic_data(num_nodes=100, num_features=16, num_classes=4):
        # Use modern Flax NNX Rngs shorthand methods
        rngs = nnx.Rngs(42)

        # Random features
        x = rngs.normal((num_nodes, num_features))

        # Random edges (Erdős-Rényi graph)
        prob = 0.1
        adj = rngs.bernoulli(prob, (num_nodes, num_nodes))
        edge_index = jnp.array(jnp.where(adj)).astype(jnp.int32)

        # Random labels
        y = rngs.randint((num_nodes,), 0, num_classes)

        # Train/val/test splits using indices (JIT-friendly)
        indices = rngs.permutation(jnp.arange(num_nodes))
        train_size = int(0.6 * num_nodes)
        val_size = int(0.8 * num_nodes)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:val_size]
        test_indices = indices[val_size:]

        # Create basic data object
        data = Data(x=x, edge_index=edge_index, y=y)
        return data, train_indices, val_indices, test_indices

    # Create data
    data, train_indices, val_indices, test_indices = create_synthetic_data()

    # Initialize model and optimizer
    model = BasicGCN(16, 32, 4, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    # Training function
    @nnx.jit
    def train_step(model, optimizer, data, train_indices):
        # Ensure model is in training mode
        model.train()

        def loss_fn(model):
            logits = model(data.x, data.edge_index)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits[train_indices],
                data.y[train_indices]
            ).mean()
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Evaluation function
    @nnx.jit
    def evaluate(model, data, indices):
        # Create evaluation model that shares weights
        eval_model = nnx.merge(*nnx.split(model))
        eval_model.eval()

        logits = eval_model(data.x, data.edge_index)
        preds = jnp.argmax(logits, axis=-1)
        accuracy = (preds[indices] == data.y[indices]).mean()
        return accuracy

    # Training loop
    for epoch in range(200):
        loss = train_step(model, optimizer, data, train_indices)

        if epoch % 20 == 0:
            train_acc = evaluate(model, data, train_indices)
            val_acc = evaluate(model, data, val_indices)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

    # Final evaluation
    test_acc = evaluate(model, data, test_indices)
    print(f"Test Accuracy: {test_acc:.3f}")

Graph Classification Example
-----------------------------

Example for classifying entire graphs:

.. code-block:: python

    from jraphx.data import Batch
    from jraphx.nn.pool import global_mean_pool

    class GraphClassifier(nnx.Module):
        def __init__(self, in_features, hidden_dim, num_classes, rngs):
            self.conv1 = GCNConv(in_features, hidden_dim, rngs=rngs)
            self.conv2 = GCNConv(hidden_dim, hidden_dim, rngs=rngs)
            self.conv3 = GCNConv(hidden_dim, hidden_dim, rngs=rngs)

            self.classifier = nnx.Sequential(
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
                nnx.relu,
                nnx.Dropout(0.5, rngs=rngs),
                nnx.Linear(hidden_dim, num_classes, rngs=rngs)
            )

        def __call__(self, x, edge_index, batch):
            # Graph convolutions
            x = nnx.relu(self.conv1(x, edge_index))
            x = nnx.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)

            # Global pooling
            x = global_mean_pool(x, batch)

            # Classification
            return self.classifier(x)

    # Create synthetic graph dataset
    def create_graph_dataset(num_graphs=100, num_classes=2):
        graphs = []
        for i in range(num_graphs):
            # Use modern Flax NNX patterns with different keys for each graph
            rngs = nnx.Rngs(i)
            num_nodes = rngs.randint((), 10, 30)

            x = rngs.normal((num_nodes, 16))
            prob = 0.3
            adj = rngs.bernoulli(prob, (num_nodes, num_nodes))
            edge_index = jnp.array(jnp.where(adj))

            y = rngs.randint((), 0, num_classes)

            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        return graphs

    # Create dataset
    graphs = create_graph_dataset(100, 2)
    train_graphs = graphs[:80]
    test_graphs = graphs[80:]

    # Batch graphs
    train_batch = Batch.from_data_list(train_graphs)
    test_batch = Batch.from_data_list(test_graphs)

    # Initialize model
    classifier = GraphClassifier(16, 64, 2, rngs=nnx.Rngs(0))

    # Forward pass
    logits = classifier(
        train_batch.x,
        train_batch.edge_index,
        train_batch.batch
    )
    print(f"Output shape: {logits.shape}")  # (80, 2)

Edge Prediction Example
-----------------------

Link prediction using node embeddings:

.. code-block:: python

    class LinkPredictor(nnx.Module):
        def __init__(self, in_features, hidden_dim, rngs):
            self.conv1 = GCNConv(in_features, hidden_dim, rngs=rngs)
            self.conv2 = GCNConv(hidden_dim, hidden_dim, rngs=rngs)

        def encode(self, x, edge_index):
            x = nnx.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x

        def decode(self, z, edge_index):
            # Simple dot product decoder
            src, dst = edge_index
            return (z[src] * z[dst]).sum(axis=-1)

        def __call__(self, x, edge_index, pos_edge_index, neg_edge_index=None):
            # Encode nodes
            z = self.encode(x, edge_index)

            # Decode edges
            pos_pred = self.decode(z, pos_edge_index)

            if neg_edge_index is not None:
                neg_pred = self.decode(z, neg_edge_index)
                return pos_pred, neg_pred

            return pos_pred

    # Create link prediction data
    def prepare_link_prediction_data(data, train_ratio=0.8):
        num_edges = data.edge_index.shape[1]
        num_train = int(train_ratio * num_edges)

        # Shuffle edges using modern Flax NNX patterns
        rngs = nnx.Rngs(42)
        perm = rngs.permutation(jnp.arange(num_edges))

        # Split edges
        train_edge_index = data.edge_index[:, perm[:num_train]]
        test_edge_index = data.edge_index[:, perm[num_train:]]

        # Sample negative edges
        num_neg = test_edge_index.shape[1]
        neg_edges = []
        for _ in range(num_neg):
            src = rngs.randint((), 0, data.num_nodes)
            dst = rngs.randint((), 0, data.num_nodes)
            neg_edges.append([src, dst])

        neg_edge_index = jnp.array(neg_edges).T

        return train_edge_index, test_edge_index, neg_edge_index

    # Prepare data
    train_edges, test_edges, neg_edges = prepare_link_prediction_data(data)

    # Initialize model
    link_model = LinkPredictor(data.num_node_features, 32, rngs=nnx.Rngs(0))

    # Predict links
    pos_scores, neg_scores = link_model(
        data.x, train_edges, test_edges, neg_edges
    )

    # Compute accuracy
    pos_pred = pos_scores > 0
    neg_pred = neg_scores <= 0
    accuracy = jnp.concatenate([pos_pred, neg_pred]).mean()
    print(f"Link prediction accuracy: {accuracy:.3f}")

Running the Examples
--------------------

To run these examples:

1. **Install JraphX**:

   .. code-block:: bash

      pip install jraphx

2. **Copy the code** into a Python file or Jupyter notebook

3. **Run the examples**:

   .. code-block:: bash

      python basic_examples.py

Each example is self-contained and demonstrates different aspects of JraphX:

- Graph construction and manipulation
- Building GNN models
- Training and evaluation
- Different tasks (node classification, graph classification, link prediction)

Exploring Real Examples
-----------------------

For more comprehensive and advanced examples, explore the ``examples/`` directory in the JraphX repository:

**Getting Started Examples:**
- ``gcn_jraphx.py`` - Complete GCN implementation with real datasets
- ``karate_club.py`` - Classic graph clustering example
- ``pre_built_models.py`` - Using JraphX's pre-built model library

**Advanced Examples:**
- ``gat_example.py`` - Graph Attention Networks with multi-head attention
- ``cora_planetoid.py`` - Citation network node classification
- ``nnx_transforms.py`` - Advanced JAX transformations and vectorization
- ``batch_node_prediction.py`` - Efficient batched processing

**Research Examples:**
- ``graph_saint_flickr.py`` - Large-scale graph sampling and training
- ``tempo_diffusion.py`` - Temporal graph diffusion models

These examples demonstrate production-ready code patterns, real dataset handling, and advanced JraphX features. They're perfect for understanding how to apply JraphX to your own research or projects.

Next Steps
----------

- Explore the :doc:`gnn_design` tutorial for advanced JAX integration patterns
- Check the :doc:`../modules/nn` for all available GNN layers
- Browse the repository's ``examples/`` directory for cutting-edge implementations
