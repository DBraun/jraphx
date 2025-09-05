Introduction by Example
=======================

We shortly introduce the fundamental concepts of **JraphX** through self-contained examples.

For an introduction to Graph Machine Learning, we refer the interested reader to the `Stanford CS224W: Machine Learning with Graphs <https://www.youtube.com/watch?v=JAB_plj2rbA>`__ lectures.
For an introduction to JAX, see the `JAX documentation <https://jax.readthedocs.io/>`__.

At its core, **JraphX** provides the following main features:

.. contents::
    :local:

Data Handling of Graphs
-----------------------

A graph is used to model pairwise relations (edges) between objects (nodes).
A single graph in **JraphX** is described by an instance of :class:`jraphx.data.Data`, which holds the following attributes by default:

- :obj:`data.x`: Node feature matrix with shape :obj:`[num_nodes, num_node_features]` as a JAX array
- :obj:`data.edge_index`: Graph connectivity with shape :obj:`[2, num_edges]` as a JAX array with integer dtype
- :obj:`data.edge_attr`: Edge feature matrix with shape :obj:`[num_edges, num_edge_features]` as a JAX array
- :obj:`data.y`: Target to train against (may have arbitrary shape), *e.g.*, node-level targets of shape :obj:`[num_nodes, *]` or graph-level targets of shape :obj:`[1, *]`
- :obj:`data.pos`: Node position matrix with shape :obj:`[num_nodes, num_dimensions]` for 3D point clouds

None of these attributes are required.
In fact, the :class:`~jraphx.data.Data` object is not even restricted to these attributes.
We can extend it to save the connectivity of triangles from a 3D mesh in a JAX array with shape :obj:`[3, num_faces]`.
See the :ref:`3D Mesh Graphs example <Extending-Both-Data-and-Batch-Classes>` for a complete implementation.

.. Note::
    JAX uses a functional programming paradigm where arrays are immutable. This means that operations on :class:`~jraphx.data.Data` objects return new instances rather than modifying existing ones.

We show a simple example of an unweighted and undirected graph with three nodes and four edges.
Each node contains exactly one feature:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data

    edge_index = jnp.array([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=jnp.int32)
    x = jnp.array([[-1.0], [0.0], [1.0]], dtype=jnp.float32)

    data = Data(x=x, edge_index=edge_index)
    >>> Data(edge_index=[2, 4], x=[3, 1])

.. image:: ../_figures/graph.svg
  :align: center
  :width: 300px

|

Note that :obj:`edge_index`, *i.e.* the array defining the source and target nodes of all edges, is **not** a list of index tuples.
If you want to write your indices this way, you should transpose it before passing to the data constructor:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data

    edge_index = jnp.array([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2, 1]], dtype=jnp.int32)
    x = jnp.array([[-1.0], [0.0], [1.0]], dtype=jnp.float32)

    data = Data(x=x, edge_index=edge_index.T)
    >>> Data(edge_index=[2, 4], x=[3, 1])

Although the graph has only two edges, we need to define four index tuples to account for both directions of a edge.

.. Note::
    You can print out your data object anytime and receive information about its attributes and their shapes.

Note that it is necessary that the elements in :obj:`edge_index` only hold indices in the range :obj:`{ 0, ..., num_nodes - 1}`.
This is needed as we want our final data representation to be as compact as possible, *e.g.*, we want to index the source and destination node features of the first edge :obj:`(0, 1)` via :obj:`x[0]` and :obj:`x[1]`, respectively.

Besides holding a number of node-level, edge-level or graph-level attributes, :class:`~jraphx.data.Data` provides a number of useful utility functions, *e.g.*:

.. code-block:: python

    print(data.keys())
    >>> ['x', 'edge_index']

    print(data['x'])
    >>> Array([[-1.0],
               [ 0.0],
               [ 1.0]], dtype=float32)

    for key, item in data:
        print(f'{key} found in data')
    >>> x found in data
    >>> edge_index found in data

    'edge_attr' in data
    >>> False

    data.num_nodes
    >>> 3

    data.num_edges
    >>> 4

    data.num_node_features
    >>> 1

    data.has_isolated_nodes()
    >>> False

    data.has_self_loops()
    >>> False

    data.is_directed
    >>> False

You can find a complete list of all methods at :class:`jraphx.data.Data`.

Working with JAX Arrays
-----------------------

**JraphX** is designed to work seamlessly with JAX arrays and the JAX ecosystem. Unlike PyTorch tensors, JAX arrays are immutable and operations are functional. Here are some key concepts:

JAX arrays can be created from Python lists or NumPy arrays:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data

    # Create JAX arrays for graph data
    node_features = jnp.array([[1.0, 0.5], [0.0, 1.0], [0.5, 0.0]], dtype=jnp.float32)
    edge_indices = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)

    data = Data(x=node_features, edge_index=edge_indices)
    print(data.x.shape)
    >>> (3, 2)

**JraphX** integrates well with JAX's transformation system. You can use :obj:`jax.jit` to compile functions for better performance:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jraphx.utils import degree

    def compute_node_degrees(edge_index, num_nodes):
        """Compute node degrees using JAX."""
        return degree(edge_index[1], num_nodes)

    # JIT compile with static num_nodes argument
    jit_compute_degrees = jax.jit(compute_node_degrees, static_argnums=(1,))
    degrees = jit_compute_degrees(data.edge_index, data.x.shape[0])
    print(degrees)
    >>> [1. 1. 1.]

For processing multiple graphs efficiently, you can use :obj:`jax.vmap`:

.. code-block:: python

    # Create multiple graphs
    graphs = [Data(x=jnp.ones((3, 2)), edge_index=jnp.array([[0, 1], [1, 0]]))
              for _ in range(5)]

    # Process multiple graphs in parallel
    def process_single_graph(data):
        return jnp.sum(data.x)

    # vmap over a batch of graphs
    batched_process = jax.vmap(process_single_graph)
    # results = batched_process(graph_batch)  # Requires proper batching

Mini-batches
------------

Neural networks are usually trained in a batch-wise fashion.
**JraphX** achieves parallelization over a mini-batch by creating sparse block diagonal adjacency matrices (defined by :obj:`edge_index`) and concatenating feature and target matrices in the node dimension.
This composition allows differing number of nodes and edges over examples in one batch:

.. math::

    \mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}

**JraphX** provides a :class:`jraphx.data.Batch` class that handles this concatenation process.
Let's learn about it in an example:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data, Batch
    from jraphx.nn.pool import global_mean_pool

    # Create some example graphs
    graphs = []
    for i in range(3):
        x = jnp.ones((4, 2), dtype=jnp.float32) * (i + 1)
        edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=jnp.int32)
        graphs.append(Data(x=x, edge_index=edge_index))

    # Create a batch from multiple graphs
    batch = Batch.from_data_list(graphs)
    print(batch)
    >>> Batch(batch=[12], edge_index=[2, 12], x=[12, 2])

    print(batch.num_graphs)
    >>> 3

:class:`jraphx.data.Batch` inherits from :class:`jraphx.data.Data` and contains an additional attribute called :obj:`batch`.

:obj:`batch` is a column vector which maps each node to its respective graph in the batch:

.. math::

    \mathrm{batch} = {\begin{bmatrix} 0 & \cdots & 0 & 1 & \cdots & n - 2 & n -1 & \cdots & n - 1 \end{bmatrix}}^{\top}

You can use it to, *e.g.*, average node features in the node dimension for each graph individually:

.. code-block:: python

    from jraphx.utils import scatter

    # Average node features per graph
    graph_embeddings = scatter(batch.x, batch.batch, dim_size=batch.num_graphs, dim=0, reduce='mean')
    print(graph_embeddings.shape)
    >>> (3, 2)  # 3 graphs, 2 features each

You can learn more about the internal batching procedure of **JraphX**, *e.g.*, how to modify its behavior, `here <../advanced/batching.html>`__.
For documentation of scatter operations, see :class:`jraphx.utils.scatter`.

Using Graph Convolution Layers
-------------------------------

**JraphX** provides various graph neural network layers:

.. code-block:: python

    import flax.nnx as nnx
    from jraphx.nn.conv import GCNConv, GATConv, SAGEConv

    # Initialize random number generator
    rngs = nnx.Rngs(42)

    # Graph Convolutional Network (GCN)
    gcn = GCNConv(in_features=3, out_features=16, rngs=rngs)
    out = gcn(data.x, data.edge_index)

    # Graph Attention Network (GAT)
    gat = GATConv(in_features=3, out_features=16, heads=4, rngs=rngs)
    out = gat(data.x, data.edge_index)

    # GraphSAGE
    sage = SAGEConv(in_features=3, out_features=16, rngs=rngs)
    out = sage(data.x, data.edge_index)

Building a Complete GNN Model
------------------------------

Combine multiple layers to create a complete GNN model:

.. code-block:: python

    import jax
    import flax.nnx as nnx
    from jraphx.nn.conv import GCNConv
    from jraphx.nn.pool import global_mean_pool

    class GNN(nnx.Module):
        def __init__(self, in_features, hidden_features, out_features, rngs):
            self.conv1 = GCNConv(in_features, hidden_features, rngs=rngs)
            self.conv2 = GCNConv(hidden_features, hidden_features, rngs=rngs)
            self.conv3 = GCNConv(hidden_features, out_features, rngs=rngs)
            self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

        def __call__(self, x, edge_index, batch=None):
            # First GCN layer
            x = self.conv1(x, edge_index)
            x = nnx.relu(x)
            x = self.dropout(x)

            # Second GCN layer
            x = self.conv2(x, edge_index)
            x = nnx.relu(x)
            x = self.dropout(x)

            # Third GCN layer
            x = self.conv3(x, edge_index)

            # Global pooling (for graph-level prediction)
            if batch is not None:
                x = global_mean_pool(x, batch)

            return x

    # Create model
    model = GNN(in_features=3, hidden_features=64, out_features=10, rngs=nnx.Rngs(42))

    # Forward pass
    output = model(data.x, data.edge_index)

Model Inspection with ``nnx.tabulate``
---------------------------------------

**JraphX** leverages NNX's model inspection for transparent development:

.. code-block:: python

    from flax import nnx
    from jraphx.nn.models import GAT

    # Create model
    model = GAT(in_features=32, hidden_features=64, out_features=16,
               heads=4, num_layers=2, rngs=nnx.Rngs(42))
    x = jnp.ones((50, 32))
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

    # Inspect complete model structure and parameters
    print(nnx.tabulate(model, x, edge_index, depth=2))

This shows layer hierarchy, parameter counts, input/output shapes, and memory usage - essential for understanding complex GNN architectures before training.

Train/Eval Modes
-----------------

**NNX** provides efficient train/eval mode handling for models with dropout or batch normalization:

.. code-block:: python

    from jraphx.nn.models import GraphSAGE

    # Create model with dropout
    model = GraphSAGE(in_features=16, hidden_features=32, out_features=8,
                     num_layers=2, dropout_rate=0.5, rngs=nnx.Rngs(42))
    model.train()  # Set to training mode

    # Create evaluation model that shares weights
    eval_model = nnx.merge(*nnx.split(model))  # Same weights, different behavior
    eval_model.eval()  # Set to evaluation mode

    # Both models share weights but behave differently
    train_out = model(x, edge_index)      # Uses dropout
    eval_out = eval_model(x, edge_index)  # No dropout

    # Weights stay synchronized automatically - no copying needed!
    print("Weights shared:", jnp.allclose(
        model.convs[0].linear.kernel.value,
        eval_model.convs[0].linear.kernel.value
    ))
    >>> Weights shared: True

For more details, see the Flax documentation for `nnx.Module.train() <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module.train>`_ and `nnx.Module.eval() <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module.eval>`_.

Training a GNN
--------------

Here's a simple training loop example:

.. code-block:: python

    import optax
    from jraphx.data import DataLoader

    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.01), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, data, labels):
        # Ensure model is in training mode
        model.train()

        def loss_fn(model):
            logits = model(data.x, data.edge_index)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Training loop
    for epoch in range(100):
        loss = train_step(model, optimizer, data, labels)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

Data Preprocessing with JAX
----------------------------

**JraphX** leverages JAX's functional programming approach for data preprocessing. You can create pure functions to preprocess your data:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jraphx.data import Data
    from jraphx.utils import add_self_loops

    @jax.jit
    def preprocess_graph(data):
        """Add self-loops and normalize features."""
        # Add self-loops
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])

        # Normalize node features
        x_normalized = data.x / jnp.linalg.norm(data.x, axis=1, keepdims=True)

        return data.replace(x=x_normalized, edge_index=edge_index)

    # Apply preprocessing
    original_data = Data(x=jnp.ones((3, 2)), edge_index=jnp.array([[0, 1], [1, 2]]))
    processed_data = preprocess_graph(original_data)

For more complex preprocessing pipelines, you can compose functions:

.. code-block:: python

    def add_positional_encoding(data, rngs, dim=16):
        """Add random positional encoding to nodes."""
        pos_enc = rngs.normal((data.x.shape[0], dim))  # Flax 0.11.2 shorthand method!
        x_with_pos = jnp.concatenate([data.x, pos_enc], axis=1)
        return data.replace(x=x_with_pos)

    def preprocessing_pipeline(data, rngs):
        """Full preprocessing pipeline."""
        data = preprocess_graph(data)
        data = add_positional_encoding(data, rngs)
        return data

    # Apply full pipeline with random number generator
    rngs = nnx.Rngs(42)  # Can also use: rngs = nnx.Rngs(0, params=1)
    final_data = preprocessing_pipeline(original_data, rngs)

Learning Methods on Graphs
--------------------------

After learning about data handling and preprocessing in **JraphX**, it's time to implement our first graph neural network!

We will use a simple GCN layer implemented with JAX and Flax NNX.
For a high-level explanation on GCN, have a look at its `blog post <http://tkipf.github.io/graph-convolutional-networks/>`_.

Let's create some example graph data:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data

    # Create a simple graph with 4 nodes, 3 features per node, 3 classes
    x = jnp.array([[1.0, 0.5, 0.2], [0.8, 1.0, 0.1], [0.3, 0.7, 1.0], [0.9, 0.2, 0.8]], dtype=jnp.float32)
    edge_index = jnp.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=jnp.int32)  # Undirected edges
    y = jnp.array([0, 0, 1, 1], dtype=jnp.int32)  # Node labels

    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

Now let's implement a two-layer GCN using Flax NNX:

.. code-block:: python

    import jax.numpy as jnp
    from flax import nnx
    from jraphx.nn.conv import GCNConv

    class GCN(nnx.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int, *, rngs: nnx.Rngs):
            self.conv1 = GCNConv(in_features, hidden_features, rngs=rngs)
            self.conv2 = GCNConv(hidden_features, out_features, rngs=rngs)
            self.dropout = nnx.Dropout(0.1, rngs=rngs)

        def __call__(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = nnx.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)

            return nnx.log_softmax(x, axis=1)

    # Create model
    model = GCN(in_features=3, hidden_features=16, out_features=3, rngs=nnx.Rngs(42))

The model defines two :class:`~jraphx.nn.conv.GCNConv` layers which get called in sequence.
Note that the non-linearity is not integrated in the :obj:`conv` calls and hence needs to be applied afterwards (consistent with **JraphX** design).
Here, we use ReLU as our intermediate non-linearity and output a log-softmax distribution over classes.

Let's create a simple training function using JAX:

.. code-block:: python

    import optax

    def loss_fn(model, data, train_mask):
        """Compute cross-entropy loss on training nodes."""
        logits = model(data)
        # Select only training nodes
        train_logits = logits[train_mask]
        train_labels = data.y[train_mask]
        return optax.softmax_cross_entropy_with_integer_labels(train_logits, train_labels).mean()

    # Setup optimizer
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    # Training loop
    train_mask = jnp.array([True, True, False, False])  # First 2 nodes for training
    test_mask = jnp.array([False, False, True, True])   # Last 2 nodes for testing

    @nnx.jit
    def train_step(model, optimizer, data, train_mask):
        def loss_fn_inner(model):
            return loss_fn(model, data, train_mask)

        loss, grads = nnx.value_and_grad(loss_fn_inner)(model)
        optimizer.update(model, grads)
        return loss

    # Train for a few epochs
    model.train()
    for epoch in range(200):
        loss = train_step(model, optimizer, data, train_mask)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

Finally, we can evaluate our model:

.. code-block:: python

    @nnx.jit
    def evaluate(model, data, test_mask):
        """Evaluate model accuracy on test nodes."""
        logits = model(data)
        pred = jnp.argmax(logits, axis=1)
        correct = jnp.sum(pred[test_mask] == data.y[test_mask])
        accuracy = correct / jnp.sum(test_mask)
        return accuracy

    model.eval()
    test_accuracy = evaluate(model, data, test_mask)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    >>> Test Accuracy: 0.5000  # Small dataset, results may vary


This is all it takes to implement your first graph neural network with **JraphX**!
The key advantages of using JAX/Flax NNX are automatic differentiation, JIT compilation for speed, and functional programming patterns.
The easiest way to learn more about Graph Neural Networks is to browse :mod:`jraphx.nn` and experiment with different layer combinations.

Exercises
---------

1. What does :obj:`edge_index.T` do in JAX? How is it different from PyTorch's :obj:`edge_index.t().contiguous()`?

2. Create a function that generates a random graph with :obj:`n` nodes and :obj:`m` edges using JAX arrays. Make sure the function is JIT-compilable.

3. What does each number of the following output mean?

   .. code-block:: python

       print(batch)
       >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

4. Implement a preprocessing function using :obj:`@jax.jit` that adds self-loops to a graph and normalizes node features. Test it on a simple graph.

5. Create a batched version of the GCN model that can process multiple graphs simultaneously using :obj:`nnx.vmap`.
