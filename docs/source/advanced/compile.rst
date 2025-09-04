JAX Compilation and XLA
=======================

JAX automatically compiles your **JraphX** code to optimized XLA (Accelerated Linear Algebra) programs, providing significant performance improvements.
Unlike PyTorch's :meth:`torch.compile`, JAX compilation is built into the core of the framework and works seamlessly with **JraphX** models.

XLA Compilation Benefits
------------------------

XLA compilation in JAX provides several advantages for **JraphX** models:

- **Automatic optimization**: XLA optimizes the entire computation graph
- **Cross-platform**: Same optimizations work on CPU, GPU, and TPU
- **Operator fusion**: Combines multiple operations for better memory usage
- **Vectorization**: Automatic SIMD optimization

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jraphx.nn.models import GCN
    from flax import nnx

    # Create model - XLA will optimize this automatically when JIT-compiled
    model = GCN(
        in_features=64,
        hidden_features=128,
        out_features=32,
        num_layers=4,
        rngs=nnx.Rngs(42)
    )

    # JIT compilation triggers XLA optimization
    @jax.jit
    def optimized_forward(model, x, edge_index):
        return model(x, edge_index)

    # XLA optimizes the entire computation graph
    x = jnp.ones((1000, 64))
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])
    output = optimized_forward(model, x, edge_index)

Compilation vs. Eager Mode
--------------------------

**JraphX** models can run in both eager mode (for debugging) and compiled mode (for production):

.. code-block:: python

    # Eager mode - good for debugging
    def debug_model(model, x, edge_index):
        print(f"Input shape: {x.shape}")
        output = model(x, edge_index)
        print(f"Output shape: {output.shape}")
        return output

    # Compiled mode - good for production
    @jax.jit
    def production_model(model, x, edge_index):
        return model(x, edge_index)

    # Use debug mode during development
    debug_output = debug_model(model, x, edge_index)

    # Switch to compiled mode for performance
    prod_output = production_model(model, x, edge_index)

Advanced Compilation Features
-----------------------------

**Static argument handling:**

.. code-block:: python

    # Some arguments can be marked as static for better optimization
    @jax.jit
    def gnn_with_static_args(model, x, edge_index, num_layers=3):
        # Process with different numbers of layers
        for _ in range(num_layers):
            x = model.layers[0](x, edge_index)  # Simplified example
        return x

    # Use static_argnums to mark compile-time constants
    @jax.jit(static_argnums=(3,))  # num_layers is static
    def optimized_gnn_with_static(model, x, edge_index, num_layers):
        return gnn_with_static_args(model, x, edge_index, num_layers)

**Donation optimization:**

.. code-block:: python

    # Use donate_argnums to optimize memory usage
    @jax.jit(donate_argnums=(1,))  # Donate the 'x' argument
    def memory_efficient_gnn(model, x, edge_index):
        # JAX can reuse memory from 'x' for the output
        return model(x, edge_index)

Debugging Compiled Code
-----------------------

When debugging JIT-compiled **JraphX** models, you can:

1. **Disable JIT temporarily**:

.. code-block:: python

    with jax.disable_jit():
        output = optimized_forward(model, x, edge_index)  # Runs in eager mode

2. **Use JAX debugging tools**:

.. code-block:: python

    # Print intermediate values (only works in eager mode)
    def debug_forward(model, x, edge_index):
        x = model.layers[0](x, edge_index)
        jax.debug.print("After layer 0: {}", x.shape)
        x = model.layers[1](x, edge_index)
        jax.debug.print("After layer 1: {}", x.shape)
        return x

3. **Check compilation status**:

.. code-block:: python

    # See the compiled computation graph
    compiled_fn = jax.jit(production_model)
    print(compiled_fn.lower(model, x, edge_index).compile().as_text())

Performance Comparison
----------------------

Here's how **JraphX** with JAX compilation compares to other approaches:

.. code-block:: python

    import time
    import jax

    # Measure compilation overhead (one-time cost)
    start = time.time()
    jit_fn = jax.jit(lambda m, x, e: m(x, e))
    _ = jit_fn(model, x, edge_index)  # Compilation happens here
    compile_time = time.time() - start
    print(f"Compilation time: {compile_time:.2f}s")

    # Measure runtime performance
    start = time.time()
    for _ in range(100):
        _ = model(x, edge_index).block_until_ready()  # Eager mode
    eager_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = jit_fn(model, x, edge_index).block_until_ready()  # Compiled mode
    jit_time = time.time() - start

    print(f"Eager mode: {eager_time:.3f}s")
    print(f"JIT mode: {jit_time:.3f}s")
    print(f"Speedup: {eager_time / jit_time:.2f}x")

For more information on JAX compilation and XLA, see the `JAX compilation guide <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`__.
