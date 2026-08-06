JAX JIT Compilation
===================

JAX provides Just-In-Time (JIT) compilation through XLA to optimize and accelerate your **JraphX** models.
JIT compilation can provide significant performance improvements by optimizing the entire computation graph.
If you are unfamiliar with JAX JIT, we recommend reading the official "`JAX JIT tutorial <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_" first.

JIT-Compiling GNN Models
------------------------

All **JraphX** layers and models are designed to be JIT-compatible out of the box.

.. warning::

    Use :func:`nnx.jit`, not :func:`jax.jit`, whenever the compiled function mutates
    module state. NNX modules are pytrees, so :func:`jax.jit` accepts a model and an
    optimizer without complaint, but it traces a *copy* of their state and discards
    every in-place update when the function returns. A training step wrapped in
    :func:`jax.jit` therefore raises no error, prints a perfectly plausible loss, and
    never changes a single parameter -- the loss is bit-identical on every epoch and
    ``optimizer.step`` stays at zero. The same silence hides a frozen
    :class:`nnx.Dropout` key, which reuses one mask forever, and frozen
    :class:`~jraphx.nn.norm.BatchNorm` running statistics.

    :func:`nnx.jit` understands NNX state and propagates the mutation back out.
    :func:`jax.jit` is only safe for a genuinely pure function -- a forward pass on a
    model in evaluation mode, with no parameter update and no RNG draw.

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
    @nnx.jit
    def predict(model, x, edge_index):
        return model(x, edge_index)

    # Evaluation mode makes dropout deterministic, so this really is a pure function
    model.eval()

    # First call compiles, subsequent calls are fast
    predictions = predict(model, data.x, data.edge_index)
    print(f"Predictions shape: {predictions.shape}")

JIT-Compiling Training Steps
----------------------------

For optimal performance, JIT-compile your entire training step. This step updates the
parameters, so it must be wrapped in :func:`nnx.jit`:

.. code-block:: python

    import optax

    # Setup optimizer
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

    model.train()

    @nnx.jit
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

    # This layer is automatically JIT-compatible. Build it outside the compiled
    # function: a module constructed inside the trace is rebuilt on every call and its
    # freshly initialized parameters are thrown away when the function returns.
    layer = CustomGNNLayer(16, 32, rngs=nnx.Rngs(42))

    @nnx.jit
    def forward_with_custom_layer(layer, x, edge_index):
        return layer(x, edge_index)

Operations That Need Static Sizes
---------------------------------

A few operations produce a data-dependent number of rows, which XLA cannot express:

* :func:`~jraphx.nn.pool.global_add_pool` and friends need ``size=<num_graphs>`` when
  ``batch`` is traced, otherwise they raise a :obj:`ValueError`.
* :class:`~jraphx.nn.norm.GraphNorm` and :class:`~jraphx.nn.norm.LayerNorm` with
  ``mode="graph"`` need ``batch_size=<num_graphs>`` under a trace.
* :class:`~jraphx.nn.pool.TopKPooling` and :class:`~jraphx.nn.pool.SAGPooling` keep a
  data-dependent number of nodes and cannot be traced.
* :func:`~jraphx.utils.coalesce` and :func:`~jraphx.utils.to_undirected` return a
  data-dependent number of edges and cannot be traced.
* :class:`~jraphx.nn.conv.GCNConv` with ``cached=True`` must have its cache filled by an
  eager ``precompute_norm()`` call before the traced forward pass.

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
    fast_predict = nnx.jit(slow_predict)

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

1. **JIT the training step with** :func:`nnx.jit`: Compile the entire step for maximum
   benefit, and reach for :func:`jax.jit` only when nothing is mutated
2. **Warm up on dummy data**: Compile before timing-critical sections
3. **Static shapes**: Use fixed-size arrays when possible for better optimization
4. **Batch processing**: JIT works especially well with batched operations
5. **Avoid Python loops**: Use :func:`jax.lax.scan` or :func:`nnx.vmap` instead

.. code-block:: python

    # Good: JIT-friendly batch processing
    @nnx.jit
    def process_batch(model, batch_x, batch_edge_index):
        return nnx.vmap(model)(batch_x, batch_edge_index)

    # Better: Use JraphX Batch for variable-size graphs
    @nnx.jit
    def process_jraphx_batch(model, batch):
        return model(batch.x, batch.edge_index)

Common Pitfalls
---------------

- **Mutating state under** :func:`jax.jit`: The most costly pitfall, because it is
  silent. Parameter updates, RNG draws and running statistics do not escape a
  :func:`jax.jit` boundary; use :func:`nnx.jit` for anything that writes to a module
- **Dynamic shapes**: Avoid operations that change array shapes based on data
- **Python conditionals**: Use :func:`jnp.where` instead of :obj:`if` statements
- **Global state**: Avoid modifying global variables inside JIT functions
- **Device transfers**: Minimize data movement between devices within JIT functions

A quick way to catch a silently frozen training step is to assert that it actually
moves:

.. code-block:: python

    before = jax.tree.leaves(nnx.state(model, nnx.Param))
    loss = train_step(model, optimizer, data.x, data.edge_index, targets, train_indices)
    after = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert any((a != b).any() for a, b in zip(before, after)), "no parameter changed"

For more information on JAX JIT compilation, see the `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`__.
