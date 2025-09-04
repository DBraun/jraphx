Advanced Mini-Batching with JAX
================================

The creation of mini-batching is crucial for letting the training of a deep learning model scale to huge amounts of data.
Instead of processing examples one-by-one, a mini-batch groups a set of examples into a unified representation where it can efficiently be processed in parallel.
In the image or language domain, this procedure is typically achieved by rescaling or padding each example into a set to equally-sized shapes, and examples are then grouped in an additional dimension.
The length of this dimension is then equal to the number of examples grouped in a mini-batch and is typically referred to as the :obj:`batch_size`.

Since graphs are one of the most general data structures that can hold *any* number of nodes or edges, the two approaches described above are either not feasible or may result in a lot of unnecessary memory consumption.
In **JraphX**, we adopt the same approach as PyTorch Geometric to achieve parallelization across a number of examples.
Here, adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple isolated subgraphs), and node and target features are simply concatenated in the node dimension, *i.e.*

.. math::

    \mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}.

This procedure has some crucial advantages over other batching procedures:

1. GNN operators that rely on a message passing scheme do not need to be modified since messages still cannot be exchanged between two nodes that belong to different graphs.

2. There is no computational or memory overhead.
   For example, this batching procedure works completely without any padding of node or edge features.
   Note that there is no additional memory overhead for adjacency matrices since they are saved in a sparse fashion holding only non-zero entries, *i.e.*, the edges.

**JraphX** automatically takes care of batching multiple graphs into a single giant graph with the help of the :class:`jraphx.data.Batch` class.

Basic Batching in JraphX
------------------------

**JraphX** uses the :class:`jraphx.data.Batch` class to handle batching multiple graphs. Here's how it works:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data, Batch

    # Create individual graphs
    graph1 = Data(
        x=jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 nodes
        edge_index=jnp.array([[0, 1], [1, 0]])   # 2 edges
    )

    graph2 = Data(
        x=jnp.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),  # 3 nodes
        edge_index=jnp.array([[0, 1, 2], [1, 2, 0]])          # 3 edges
    )

    # Create batch
    batch = Batch.from_data_list([graph1, graph2])

    print(f"Batch info:")
    print(f"  Total nodes: {batch.num_nodes}")      # 5 nodes total
    print(f"  Total edges: {batch.num_edges}")      # 5 edges total
    print(f"  Num graphs: {batch.num_graphs}")      # 2 graphs
    print(f"  Node features: {batch.x.shape}")     # [5, 2]
    print(f"  Edge indices: {batch.edge_index.shape}")  # [2, 5]
    print(f"  Batch vector: {batch.batch}")        # [0, 0, 1, 1, 1]

The :obj:`edge_index` tensor is automatically incremented by the cumulated number of nodes of all graphs that got batched before the currently processed graph, and edge indices are concatenated in the second dimension.
All other arrays are concatenated in the first dimension.

JAX-Specific Batching Benefits
------------------------------

**JraphX** batching integrates seamlessly with JAX transformations:

.. code-block:: python

    import jax
    from jraphx.nn.conv import GCNConv
    from flax import nnx

    # Create model
    model = GCNConv(2, 8, rngs=nnx.Rngs(42))

    # Process batch with JIT compilation (extract arrays first)
    @jax.jit
    def process_batch(model, x, edge_index):
        return model(x, edge_index)

    # Efficient batch processing
    batch_output = process_batch(model, batch.x, batch.edge_index)
    print(f"Batch output shape: {batch_output.shape}")  # [5, 8]

    # For multiple batches, process arrays directly
    def process_graph_list(model, graph_list):
        """Process a list of graphs efficiently."""
        batch = Batch.from_data_list(graph_list)
        return process_batch(model, batch.x, batch.edge_index)

In what follows, we present a few use-cases where custom batching behavior might be necessary.

Graph Matching with Paired Data
-------------------------------

For applications like graph matching, you may need to work with pairs of graphs. **JraphX** handles this using separate :class:`Data` objects and functional composition:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data, Batch

    # Create source and target graphs separately
    source_graph = Data(
        x=jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 nodes
        edge_index=jnp.array([[0, 1], [1, 0]])
    )

    target_graph = Data(
        x=jnp.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),  # 3 nodes
        edge_index=jnp.array([[0, 1, 2], [1, 2, 0]])
    )

    # For graph matching, process separately then combine results
    def process_graph_pair(model, source, target):
        source_embedding = model(source.x, source.edge_index)
        target_embedding = model(target.x, target.edge_index)
        return source_embedding, target_embedding

    # Use vmap to process multiple pairs efficiently
    @nnx.vmap
    def batch_process_pairs(model, source_batch, target_batch):
        return process_graph_pair(model, source_batch, target_batch)

Graph-Level Attributes
----------------------

For graph-level properties (like graph classification targets), **JraphX** handles this using functional processing and pooling:

.. code-block:: python

    from jraphx.nn.pool import global_mean_pool

    # Create graphs with graph-level targets
    graphs_with_targets = []
    rngs = nnx.Rngs(0, targets=1)  # Use Flax 0.11.2 shorthand

    for i in range(10):
        x = rngs.normal((5, 16))  # Node features
        edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])  # Simple cycle
        target = rngs.targets.normal((7,))  # Graph-level target

        graph = Data(x=x, edge_index=edge_index)
        graphs_with_targets.append((graph, target))

    # Batch processing with graph-level targets
    def process_graph_batch_with_targets(model, graphs_and_targets):
        graphs, targets = zip(*graphs_and_targets)

        # Create batch for graphs
        batch = Batch.from_data_list(graphs)

        # Process batch
        node_embeddings = model(batch.x, batch.edge_index)

        # Pool to graph-level
        graph_embeddings = global_mean_pool(node_embeddings, batch.batch)

        # Stack targets to create [num_graphs, target_dim]
        targets_array = jnp.stack(targets)

        return graph_embeddings, targets_array

Memory-Efficient Large Batch Processing
---------------------------------------

For very large datasets, **JraphX** supports memory-efficient batch processing using JAX transformations:

.. code-block:: python

    def create_large_dataset_processor(model, batch_size=32):
        """Create a memory-efficient processor for large graph datasets."""

        def process_batch_arrays(model, x, edge_index, batch_vector, targets, batch_size):
            """Process batch arrays with pooling."""
            predictions = model(x, edge_index)
            graph_preds = global_mean_pool(predictions, batch_vector, size=batch_size)

            # Compute loss
            loss = jnp.mean((graph_preds - targets) ** 2)
            return loss, graph_preds

        # JIT compile with static batch size for efficient pooling
        jit_process = jax.jit(process_batch_arrays, static_argnums=(5,))

        def process_dataset(dataset_graphs, dataset_targets):
            """Process entire dataset in memory-efficient batches."""
            total_loss = 0.0
            num_batches = len(dataset_graphs) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                batch_graphs = dataset_graphs[start_idx:end_idx]
                batch_targets = jnp.stack(dataset_targets[start_idx:end_idx])

                # Create batch and extract arrays for JIT
                batch = Batch.from_data_list(batch_graphs)
                loss, _ = jit_process(
                    model, batch.x, batch.edge_index, batch.batch,
                    batch_targets, len(batch_graphs)
                )
                total_loss += loss

            return total_loss / num_batches

        return process_dataset

    # Usage - this works!
    processor = create_large_dataset_processor(model, batch_size=64)
    avg_loss = processor(large_graph_list, large_target_list)

Vmap for Fixed-Size Graphs
------------------------------

When working with **fixed-size** graphs, you can use :obj:`nnx.vmap` for even more efficient batch processing:

.. code-block:: python

    # For graphs with the same number of nodes
    fixed_size_graphs = []
    for i in range(100):
        x = jnp.ones((10, 16))  # All graphs have 10 nodes
        edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])  # Same connectivity
        fixed_size_graphs.append(Data(x=x, edge_index=edge_index))

    # Stack into arrays for vmap processing
    stacked_x = jnp.stack([g.x for g in fixed_size_graphs])          # [100, 10, 16]
    stacked_edges = jnp.stack([g.edge_index for g in fixed_size_graphs])  # [100, 2, 3]

    # Use vmap for parallel processing
    @nnx.vmap
    def process_fixed_graphs(x, edge_index):
        return model(x, edge_index)

    # Process all graphs in parallel
    all_outputs = process_fixed_graphs(stacked_x, stacked_edges)  # [100, 10, output_dim]

This approach is extremely efficient for datasets where all graphs have the same structure, as it can leverage JAX's vectorization optimizations without the overhead of index remapping that batching requires.
