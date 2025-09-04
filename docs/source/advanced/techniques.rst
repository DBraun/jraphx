Advanced Techniques
===================

This guide covers advanced features and techniques for optimizing JraphX applications.

Memory-Efficient Training
-------------------------

Using JAX Scan for Sequential Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large graphs or long training sequences, use ``jax.lax.scan`` to reduce memory consumption:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jraphx.data import Data, Batch

   def efficient_training_loop(model, optimizer, data_list, num_epochs):
       """Memory-efficient training using scan."""

       def epoch_step(carry, epoch_data):
           model, optimizer = carry
           epoch_idx, batch_data = epoch_data

           def loss_fn(model):
               logits = model(batch_data.x, batch_data.edge_index)
               loss = optax.softmax_cross_entropy_with_integer_labels(
                   logits, batch_data.y
               ).mean()
               return loss

           loss, grads = nnx.value_and_grad(loss_fn)(model)
           optimizer.update(model, grads)

           return (model, optimizer), loss

       # Prepare data for all epochs
       all_epochs_data = []
       for epoch in range(num_epochs):
           for i, data in enumerate(data_list):
               all_epochs_data.append((epoch, data))

       # Run training with scan
       (model, optimizer), losses = jax.lax.scan(
           epoch_step,
           (model, optimizer),
           jnp.array(all_epochs_data)
       )

       return model, optimizer, losses

Gradient Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

For very deep GNN models, use gradient checkpointing to trade compute for memory:

.. code-block:: python

   from jax import checkpoint

   class DeepGNN(nnx.Module):
       def __init__(self, num_layers, hidden_dim, rngs):
           self.layers = [
               GCNConv(hidden_dim, hidden_dim, rngs=rngs)
               for _ in range(num_layers)
           ]

       def __call__(self, x, edge_index):
           for i, layer in enumerate(self.layers):
               # Checkpoint every other layer
               if i % 2 == 0:
                   x = checkpoint(layer)(x, edge_index)
               else:
                   x = layer(x, edge_index)
               x = nnx.relu(x)
           return x

Vectorized Graph Processing with vmap
--------------------------------------

Process multiple graphs in parallel using JAX's ``vmap``:

.. code-block:: python

   import jax
   from jraphx.data.vmap_batch import pad_graph_data

   def process_single_graph(data, model):
       """Process a single graph."""
       return model(data.x, data.edge_index)

   # Vectorize over a batch of graphs
   process_batch = nnx.vmap(process_single_graph, in_axes=(0, None))

   # Pad graphs to same size for vmap
   padded_graphs = pad_graph_data(graph_list, max_nodes=100, max_edges=200)

   # Process all graphs in parallel
   outputs = process_batch(padded_graphs, model)

Custom vmap Patterns
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def custom_vmap_aggregation(graphs, model):
       """Custom vmap pattern for graph aggregation."""

       # Define per-graph operation
       def per_graph_op(graph):
           node_features = model(graph.x, graph.edge_index)
           # Custom aggregation
           graph_feature = node_features.mean(axis=0)
           return graph_feature

       # Vectorize and apply
       vmapped_op = nnx.vmap(per_graph_op)
       graph_features = vmapped_op(graphs)

       # Further processing on all graphs
       return graph_features.mean(axis=0)

Custom Message Passing Implementations
---------------------------------------

Implementing Edge-Conditioned Convolutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.nn.conv import MessagePassing
   import flax.nnx as nnx

   class EdgeConditionedConv(MessagePassing):
       """Message passing with edge features."""

       def __init__(self, in_features, out_features, edge_dim, rngs):
           super().__init__(aggr='mean')
           self.node_mlp = nnx.Sequential(
               nnx.Linear(in_features * 2 + edge_dim, out_features, rngs=rngs),
               nnx.relu,
               nnx.Linear(out_features, out_features, rngs=rngs)
           )

       def message(self, x_i, x_j, edge_attr):
           # Concatenate source, target, and edge features
           msg = jnp.concatenate([x_i, x_j, edge_attr], axis=-1)
           return self.node_mlp(msg)

       def __call__(self, x, edge_index, edge_attr):
           return self.propagate(
               edge_index, x=x, edge_attr=edge_attr
           )

Implementing Attention Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomAttentionConv(MessagePassing):
       """Custom attention-based message passing."""

       def __init__(self, in_features, out_features, heads=4, rngs=None):
           super().__init__(aggr='add')
           self.heads = heads
           self.out_features = out_features

           self.W_q = nnx.Linear(in_features, heads * out_features, rngs=rngs)
           self.W_k = nnx.Linear(in_features, heads * out_features, rngs=rngs)
           self.W_v = nnx.Linear(in_features, heads * out_features, rngs=rngs)

       def message(self, x_i, x_j, edge_index_i, size_i):
           # Multi-head attention
           Q = self.W_q(x_i).reshape(-1, self.heads, self.out_features)
           K = self.W_k(x_j).reshape(-1, self.heads, self.out_features)
           V = self.W_v(x_j).reshape(-1, self.heads, self.out_features)

           # Compute attention scores
           scores = (Q * K).sum(axis=-1) / jnp.sqrt(self.out_features)
           alpha = nnx.softmax(scores, axis=0)

           # Apply attention to values
           return (alpha[..., None] * V).reshape(-1, self.heads * self.out_features)

Performance Optimization
------------------------


Efficient Scatter Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jraphx.utils.scatter import scatter_sum, scatter_max

   def efficient_aggregation(src, index, dim_size):
       """Combine multiple scatter operations efficiently."""

       # Compute multiple aggregations in single pass
       sum_result = scatter_sum(src, index, dim_size=dim_size)
       max_result = scatter_max(src, index, dim_size=dim_size)

       # Avoid redundant computations
       mean_result = sum_result / scatter_sum(
           jnp.ones_like(src), index, dim_size=dim_size
       )

       return {
           'sum': sum_result,
           'max': max_result,
           'mean': mean_result
       }

Working with Dynamic Graphs
----------------------------

Handling Variable-Size Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def process_dynamic_graphs(graphs):
       """Process graphs with varying sizes."""

       def process_single(graph):
           # Pad to maximum size if needed
           max_nodes = 1000
           current_nodes = graph.x.shape[0]

           if current_nodes < max_nodes:
               pad_size = max_nodes - current_nodes
               x_padded = jnp.pad(
                   graph.x,
                   ((0, pad_size), (0, 0)),
                   mode='constant'
               )
               mask = jnp.concatenate([
                   jnp.ones(current_nodes),
                   jnp.zeros(pad_size)
               ])
           else:
               x_padded = graph.x[:max_nodes]
               mask = jnp.ones(max_nodes)

           return x_padded, mask

       # Process each graph
       processed = [process_single(g) for g in graphs]
       return processed

Dynamic Edge Construction
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def construct_knn_graph(x, k=10):
       """Dynamically construct k-NN graph from node features."""

       # Compute pairwise distances
       dist_matrix = jnp.sum((x[:, None] - x[None, :]) ** 2, axis=-1)

       # Find k nearest neighbors
       _, indices = jax.lax.top_k(-dist_matrix, k)

       # Construct edge index
       num_nodes = x.shape[0]
       source = jnp.repeat(jnp.arange(num_nodes), k)
       target = indices.flatten()

       edge_index = jnp.stack([source, target])
       return edge_index

Distributed Training
--------------------

Data Parallel Training
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   from jax import pmap

   def distributed_train_step(model, optimizer, batch):
       """Single training step for data parallel training."""

       def loss_fn(model):
           logits = model(batch.x, batch.edge_index)
           loss = compute_loss(logits, batch.y)
           return loss.mean()

       loss, grads = nnx.value_and_grad(loss_fn)(model)

       # Average gradients across devices
       grads = jax.tree_map(lambda x: jax.lax.pmean(x, 'batch'), grads)

       optimizer.update(model, grads)
       return loss

   # Parallelize across devices
   parallel_train_step = pmap(distributed_train_step, axis_name='batch')

Model Parallel GNNs
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def model_parallel_gnn(x, edge_index, num_devices=2):
       """Split GNN layers across devices."""

       devices = jax.devices()[:num_devices]

       # Split layers across devices
       with jax.default_device(devices[0]):
           x = layer1(x, edge_index)
           x = layer2(x, edge_index)

       with jax.default_device(devices[1]):
           x = layer3(x, edge_index)
           x = layer4(x, edge_index)

       return x

Advanced Pooling Strategies
----------------------------

Differentiable Pooling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DiffPool(nnx.Module):
       """Differentiable pooling layer."""

       def __init__(self, in_features, ratio=0.5, rngs=None):
           self.pool_gnn = GCNConv(in_features, int(in_features * ratio), rngs=rngs)
           self.embed_gnn = GCNConv(in_features, in_features, rngs=rngs)

       def __call__(self, x, edge_index, batch=None):
           # Compute cluster assignments
           s = self.pool_gnn(x, edge_index)
           s = nnx.softmax(s, axis=-1)

           # Compute new node features
           x_pooled = s.T @ x

           # Compute new adjacency
           adj = to_dense_adj(edge_index)
           adj_pooled = s.T @ adj @ s

           # Convert back to edge index
           edge_index_pooled = to_edge_index(adj_pooled)

           return x_pooled, edge_index_pooled, s

Best Practices Summary
----------------------

1. **Memory Management**
   - Use ``jax.lax.scan`` for sequential operations
   - Apply gradient checkpointing for deep models
   - Batch graphs efficiently with padding

2. **Performance**
   - JIT compile performance-critical functions
   - Use static arguments for conditional logic
   - Leverage vmap for parallel processing

3. **Scalability**
   - Implement data parallel training with pmap
   - Use model parallelism for very large models
   - Consider dynamic batching for variable-size graphs

4. **Debugging**
   - Use ``jax.debug.print`` inside JIT-compiled functions
   - Check shapes with ``jax.debug.breakpoint``
   - Profile with ``jax.profiler``

See Also
--------

- :doc:`/modules/nn` - Neural network layer reference
- :doc:`/tutorial/gnn_design` - Advanced JAX integration tutorial
- `JAX Documentation <https://jax.readthedocs.io>`_ - JAX performance guide
