JAX Integration with JraphX
===========================

This tutorial shows how to integrate **JraphX** with JAX's transformation system for high-performance graph neural networks.

.. contents::
    :local:

JIT Compilation
---------------

All **JraphX** models support :obj:`@jax.jit` compilation for optimal performance:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jraphx.nn.models import GCN
    from jraphx.data import Data

    # Create model and data
    model = GCN(16, 32, 7, num_layers=2, rngs=nnx.Rngs(42))
    data = Data(
        x=jnp.ones((100, 16)),
        edge_index=jnp.array([[0, 1], [1, 0]])
    )

    # JIT compile for faster execution
    @jax.jit
    def predict(model, data):
        return model(data)

    # First call compiles, subsequent calls are fast
    predictions = predict(model, data)

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

    # For fixed-size graphs, use vmap directly
    @nnx.vmap
    def batch_predict(data):
        return model(data)

    # For variable-size graphs, use Batch
    batch = Batch.from_data_list(graphs)
    batch_predictions = predict(model, batch)

Training with NNX
-----------------

**JraphX** integrates seamlessly with Flax NNX for training:

.. code-block:: python

    import optax

    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    # Training step with JIT compilation
    @jax.jit
    def train_step(model, optimizer, data, targets):
        def loss_fn(model):
            predictions = model(data)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                predictions, targets
            ))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Train for several epochs
    targets = jnp.array([0, 1, 0, 1, 2])  # Node labels
    for epoch in range(100):
        loss = train_step(model, optimizer, data, targets)
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

    def create_deep_gnn_with_scan(num_layers: int, in_features: int,
                                   hidden_features: int, out_features: int):
        """Create a deep GNN using nnx.scan for memory efficiency."""

        class HiddenBlock(nnx.Module):
            """Single hidden layer block for scanning."""
            def __init__(self, rngs: nnx.Rngs):
                self.conv = GCNConv(hidden_features, hidden_features, rngs=rngs)

            def __call__(self, x, edge_index):
                x = self.conv(x, edge_index)
                x = nnx.relu(x)
                return x  # Return only x, no second output needed

        class DeepGNN(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                # Create input and output layers
                self.input_layer = GCNConv(in_features, hidden_features, rngs=rngs)
                self.output_layer = GCNConv(hidden_features, out_features, rngs=rngs)

                # Create multiple hidden layers using vmap
                num_hidden = num_layers - 2

                @nnx.split_rngs(splits=num_hidden)
                @nnx.vmap(in_axes=(0,), out_axes=0)
                def create_hidden_block(rngs: nnx.Rngs):
                    return HiddenBlock(rngs=rngs)

                self.hidden_blocks = create_hidden_block(rngs)
                self.num_hidden = num_hidden

            def __call__(self, data):
                x, edge_index = data.x, data.edge_index

                # Input layer
                x = self.input_layer(x, edge_index)
                x = nnx.relu(x)

                # Hidden layers with scan
                @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
                def forward_hidden(x, block):
                    x = block(x, edge_index)
                    return x

                x = forward_hidden(x, self.hidden_blocks)

                # Output layer
                return self.output_layer(x, edge_index)

        return DeepGNN

    # Create and use deep network
    deep_model = create_deep_gnn_with_scan(10, 16, 64, 7)(rngs=nnx.Rngs(42))
    deep_predictions = deep_model(data)

Random Number Generation with Flax 0.11.2
------------------------------------------

**Flax 0.11.2** introduces convenient shorthand methods for random number generation directly on :class:`nnx.Rngs` objects:

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

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jraphx.data import Data, Batch
    from jraphx.nn.pool import global_mean_pool

    # Create multiple training graphs using new Rngs shorthand methods
    rngs = nnx.Rngs(0, params=1)  # Separate keys for different purposes
    train_graphs = []
    for i in range(100):
        # Use Rngs shorthand methods (Flax 0.11.2 feature)
        n_nodes = rngs.randint((), 10, 50)  # Much cleaner than random.randint!
        x = rngs.params.normal((n_nodes, 16))  # Use params key for features
        # Create random edges (simplified)
        n_edges = n_nodes - 1
        edge_index = jnp.stack([
            jnp.arange(n_edges),
            jnp.roll(jnp.arange(n_edges), 1)
        ])
        train_graphs.append(Data(x=x, edge_index=edge_index))

    # Batch training function
    @jax.jit
    def train_on_batch(model, optimizer, graphs, targets):
        batch = Batch.from_data_list(graphs)

        def loss_fn(model):
            predictions = model(batch)
            # Global pooling to get graph-level predictions
            graph_preds = global_mean_pool(predictions, batch.batch)
            return jnp.mean((graph_preds - targets) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Training loop
    model_rngs = nnx.Rngs(42)  # For model initialization
    model = GCN(16, 32, 7, rngs=model_rngs)
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    target_rngs = nnx.Rngs(100)  # Separate Rngs for targets
    for epoch in range(50):
        # Sample batch of graphs
        batch_graphs = train_graphs[:32]  # Batch size 32
        batch_targets = target_rngs.normal((32, 7))  # Shorthand method!

        loss = train_on_batch(model, optimizer, batch_graphs, batch_targets)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
