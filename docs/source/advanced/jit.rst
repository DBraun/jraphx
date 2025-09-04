JAX JIT Compilation
===================

JAX provides Just-In-Time (JIT) compilation through XLA to optimize and accelerate your **JraphX** models.
JIT compilation can provide significant performance improvements by optimizing the entire computation graph.
If you are unfamiliar with JAX JIT, we recommend reading the official "`JAX JIT tutorial <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_" first.

JIT-Compiling GNN Models
------------------------

All **JraphX** layers and models are designed to be JIT-compatible out of the box.
Here's how to JIT-compile a simple GNN model:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jraphx.nn.models import GCN
    from jraphx.data import Data

    # Create model and data
    model = GCN(
        in_features=16,
        hidden_features=64,
        out_features=7,
        num_layers=3,
        rngs=nnx.Rngs(42)
    )

    data = Data(
        x=jnp.ones((100, 16)),
        edge_index=jnp.array([[0, 1, 2], [1, 2, 0]])
    )

    # JIT compile the forward pass
    @jax.jit
    def predict(model, x, edge_index):
        return model(x, edge_index)

    # First call compiles, subsequent calls are fast
    predictions = predict(model, data.x, data.edge_index)
    print(f"Predictions shape: {predictions.shape}")

JIT-Compiling Training Steps
----------------------------

For optimal performance, JIT-compile your entire training step:

.. code-block:: python

    import optax

    # Setup optimizer
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    @jax.jit
    def train_step(model, optimizer, x, edge_index, targets, train_indices):
        """JIT-compiled training step."""
        def loss_fn(model):
            predictions = model(x, edge_index)
            # Use concrete indices instead of boolean mask for JIT compatibility
            train_predictions = predictions[train_indices]
            train_targets = targets[train_indices]
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                train_predictions, train_targets
            ))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Training loop with JIT compilation
    targets = jnp.array([0, 1, 2, 0, 1, 2, 0] * 14 + [0, 1, 2])  # 100 targets
    train_indices = jnp.arange(80)  # First 80 nodes for training (concrete indices)

    for epoch in range(100):
        loss = train_step(model, optimizer, data.x, data.edge_index, targets, train_indices)
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

Custom Layer JIT Compatibility
------------------------------

When creating custom **JraphX** layers, ensure they are JIT-compatible by following these guidelines:

1. **Use only JAX operations**: Avoid Python control flow in favor of :func:`jax.lax` operations
2. **Static shapes**: Ensure array shapes are statically known when possible
3. **Pure functions**: No side effects or global state modifications

.. code-block:: python

    from jraphx.nn.conv import MessagePassing

    class CustomGNNLayer(MessagePassing):
        def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
            super().__init__(aggr='mean')
            self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

        def __call__(self, x, edge_index):
            # All operations here must be JAX-compatible
            x = self.linear(x)

            # Use JAX operations for conditionals
            x = jnp.where(x > 0, x, 0.0)  # ReLU activation

            # Standard message passing
            return self.propagate(edge_index, x)

    # This layer is automatically JIT-compatible
    @jax.jit
    def forward_with_custom_layer(x, edge_index):
        layer = CustomGNNLayer(16, 32, rngs=nnx.Rngs(42))
        return layer(x, edge_index)

Performance Benefits
--------------------

JIT compilation provides several benefits for **JraphX** models:

- **Speed**: 2-10x faster execution after compilation
- **Memory**: Optimized memory usage patterns
- **Optimization**: XLA performs advanced optimizations like operator fusion
- **Parallelization**: Automatic vectorization where possible

**Benchmarking JIT vs non-JIT:**

.. code-block:: python

    import time

    # Non-JIT version
    def slow_predict(model, x, edge_index):
        return model(x, edge_index)

    # JIT version
    fast_predict = jax.jit(slow_predict)

    # Warm up JIT (compilation happens here)
    _ = fast_predict(model, data.x, data.edge_index)

    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = slow_predict(model, data.x, data.edge_index)
    slow_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = fast_predict(model, data.x, data.edge_index)
    fast_time = time.time() - start

    print(f"Speed improvement: {slow_time / fast_time:.2f}x")

Best Practices
--------------

1. **JIT the training step**: Compile the entire training step for maximum benefit
2. **Warm up on dummy data**: Compile before timing-critical sections
3. **Static shapes**: Use fixed-size arrays when possible for better optimization
4. **Batch processing**: JIT works especially well with batched operations
5. **Avoid Python loops**: Use :func:`jax.lax.scan` or :func:`nnx.vmap` instead

.. code-block:: python

    # Good: JIT-friendly batch processing
    @jax.jit
    def process_batch(model, batch_x, batch_edge_index):
        return nnx.vmap(model)(batch_x, batch_edge_index)

    # Better: Use JraphX Batch for variable-size graphs
    @jax.jit
    def process_jraphx_batch(model, batch):
        return model(batch.x, batch.edge_index)

Common Pitfalls
---------------

- **Dynamic shapes**: Avoid operations that change array shapes based on data
- **Python conditionals**: Use :func:`jnp.where` instead of :obj:`if` statements
- **Global state**: Avoid modifying global variables inside JIT functions
- **Device transfers**: Minimize data movement between devices within JIT functions

For more information on JAX JIT compilation, see the `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`__.
