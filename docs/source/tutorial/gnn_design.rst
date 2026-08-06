JAX Integration with JraphX
===========================

This tutorial shows how to integrate **JraphX** with JAX's transformation system for high-performance graph neural networks.

.. contents::
    :local:

JIT Compilation
---------------

All **JraphX** models support :obj:`@nnx.jit` compilation for optimal performance:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jraphx.nn.models import GCN
    from jraphx.data import Data

    # Create model and data. Pass the feature sizes by keyword: `num_layers` is the
    # third positional parameter, so `GCN(16, 32, 7, num_layers=2)` would supply it
    # twice and raise a TypeError.
    model = GCN(
        in_features=16,
        hidden_features=32,
        out_features=7,
        num_layers=2,
        rngs=nnx.Rngs(42),
    )
    data = Data(
        x=jnp.ones((100, 16)),
        edge_index=jnp.array([[0, 1], [1, 0]])
    )

    # JIT compile for faster execution. Models take arrays, not a Data object.
    @nnx.jit
    def predict(model, x, edge_index):
        return model(x, edge_index)

    # First call compiles, subsequent calls are fast
    predictions = predict(model, data.x, data.edge_index)

Vectorization with vmap
-----------------------

Process multiple graphs efficiently using :obj:`nnx.vmap`:

.. code-block:: python

    from jraphx.data import Batch

    # Create batch of graphs
    graphs = [
        Data(x=jnp.ones((10, 16)), edge_index=jnp.array([[0, 1], [1, 0]])),
        Data(x=jnp.ones((15, 16)), edge_index=jnp.array([[0, 1], [1, 2]])),
    ]

    # For fixed-size graphs, use vmap directly over stacked arrays
    @nnx.vmap(in_axes=(None, 0, 0))
    def batch_predict(model, x, edge_index):
        return model(x, edge_index)

    # For variable-size graphs, use Batch: it concatenates the graphs into one
    # disjoint graph, so the ordinary forward pass handles it
    batch = Batch.from_data_list(graphs)
    batch_predictions = predict(model, batch.x, batch.edge_index)

Training with NNX
-----------------

**JraphX** integrates seamlessly with Flax NNX for training:

.. code-block:: python

    import optax

    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    # Training step with JIT compilation
    @nnx.jit
    def train_step(model, optimizer, x, edge_index, targets):
        def loss_fn(model):
            predictions = model(x, edge_index)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                predictions, targets
            ))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # One label per node, so the graph's 100 nodes need 100 targets
    targets = jnp.arange(100) % 7
    for epoch in range(100):
        loss = train_step(model, optimizer, data.x, data.edge_index, targets)
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

Train/Eval Mode Management
-----------------------------------

For train/eval mode management, see the `Introduction guide <../get_started/introduction.html#train-eval-modes>`_.

Memory-Efficient Sequential Processing
--------------------------------------

Use :obj:`nnx.scan` for memory-efficient processing of deep networks:

.. code-block:: python

    from jraphx.nn.conv import GCNConv

    class HiddenBlock(nnx.Module):
        """Single hidden layer block for scanning."""
        def __init__(self, hidden_features: int, rngs: nnx.Rngs):
            self.conv = GCNConv(hidden_features, hidden_features, rngs=rngs)

        def __call__(self, x, edge_index):
            x = self.conv(x, edge_index)
            x = nnx.relu(x)
            return x  # Return only x, no second output needed

    class DeepGNN(nnx.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int, rngs: nnx.Rngs):
            # Create input and output layers
            self.input_layer = GCNConv(in_features, hidden_features, rngs=rngs)
            self.output_layer = GCNConv(hidden_features, out_features, rngs=rngs)

            # Create multiple hidden layers using vmap
            num_hidden = num_layers - 2
            self.num_hidden = num_hidden

            if num_hidden > 0:
                @nnx.split_rngs(splits=num_hidden)
                @nnx.vmap(in_axes=(0,), out_axes=0)
                def create_hidden_block(rngs: nnx.Rngs):
                    return HiddenBlock(hidden_features, rngs=rngs)

                self.hidden_blocks = create_hidden_block(rngs)
            else:
                self.hidden_blocks = None

        def __call__(self, data):
            x, edge_index = data.x, data.edge_index

            # Input layer
            x = self.input_layer(x, edge_index)
            x = nnx.relu(x)

            # Hidden layers with scan (only if we have hidden layers)
            if self.num_hidden > 0:
                @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
                def forward_hidden(x, block):
                    x = block(x, edge_index)
                    return x

                x = forward_hidden(x, self.hidden_blocks)

            # Output layer
            return self.output_layer(x, edge_index)

    # Create and use deep network
    deep_model = DeepGNN(16, 64, 7, 10, rngs=nnx.Rngs(42))
    deep_predictions = deep_model(data.x, data.edge_index)

Random Number Generation with Flax NNX
------------------------------------------

**Flax NNX** introduces convenient shorthand methods for random number generation directly on :class:`nnx.Rngs` objects:

.. code-block:: python

    from flax import nnx

    # Create Rngs with multiple named keys
    rngs = nnx.Rngs(0, params=1, dropout=2)

    # Traditional JAX approach
    z1 = random.normal(rngs(), (2, 3))
    z2 = random.bernoulli(rngs.params(), 0.5, (10,))

    # New shorthand methods (much cleaner!)
    z1 = rngs.normal((2, 3))                   # Uses default key
    z2 = rngs.params.bernoulli(0.5, (10,))     # Uses params key
    z3 = rngs.dropout.uniform((5, 5))          # Uses dropout key

    # Example: Create random graph with different key streams
    node_features = rngs.params.normal((num_nodes, feature_dim))
    noise = rngs.dropout.normal(node_features.shape) * 0.1
    augmented_features = node_features + noise

For more details on the new randomness features, see the `Flax randomness guide <https://flax.readthedocs.io/en/latest/guides/randomness.html#jax-random-shorthand-methods>`__.

Performance Tips
----------------

1. **Always use JIT compilation** for production code
2. **Batch process multiple graphs** when possible using :obj:`nnx.vmap`
3. **Use scan for deep networks** to save memory
4. **Avoid Python loops** in favor of JAX primitives
5. **Pre-compile on dummy data** to avoid compilation during training
6. **Use Rngs shorthand methods** for cleaner random number generation

Advanced Example: Multi-Graph Training
--------------------------------------

Here's a complete example showing how to train on multiple graphs efficiently:

.. code-block:: python

    from functools import partial

    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from jraphx.data import Data, Batch
    from jraphx.nn.models import GCN
    from jraphx.nn.pool import global_mean_pool

    # Create multiple training graphs using new Rngs shorthand methods
    rngs = nnx.Rngs(0, params=1)  # Separate keys for different purposes
    train_graphs = []
    for i in range(100):
        # Use Rngs shorthand methods (Flax NNX feature)
        n_nodes = rngs.randint((), 10, 50)  # Much cleaner than random.randint!
        x = rngs.params.normal((n_nodes, 16))  # Use params key for features
        # Create random edges (simplified)
        n_edges = n_nodes - 1
        edge_index = jnp.stack([
            jnp.arange(n_edges),
            jnp.roll(jnp.arange(n_edges), 1)
        ])
        train_graphs.append(Data(x=x, edge_index=edge_index))

    # Batch training function. Collation is host-side Python work, so the Batch is
    # built outside the compiled step and only arrays cross the boundary.
    # `num_graphs` is static because pooling needs its segment count at trace time.
    @partial(nnx.jit, static_argnames=("num_graphs",))
    def train_on_batch(model, optimizer, batch, targets, num_graphs):
        def loss_fn(model):
            predictions = model(batch.x, batch.edge_index)
            # Global pooling to get graph-level predictions. `size` is required
            # here because `batch.batch` is traced inside nnx.jit.
            graph_preds = global_mean_pool(predictions, batch.batch, size=num_graphs)
            return jnp.mean((graph_preds - targets) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Training loop
    model_rngs = nnx.Rngs(42)  # For model initialization
    model = GCN(
        in_features=16, hidden_features=32, out_features=7, num_layers=2, rngs=model_rngs
    )
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    # Collate the batch and draw its targets once, outside the loop. Re-drawing the
    # targets every epoch would give the model a different objective each step, and the
    # loss would wander instead of falling.
    target_rngs = nnx.Rngs(100)  # Separate Rngs for targets
    batch_graphs = train_graphs[:32]  # Batch size 32
    batch = Batch.from_data_list(batch_graphs)
    batch_targets = target_rngs.normal((len(batch_graphs), 7))  # Shorthand method!

    for epoch in range(50):
        loss = train_on_batch(
            model, optimizer, batch, batch_targets, len(batch_graphs)
        )
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
