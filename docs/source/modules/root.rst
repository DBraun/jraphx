jraphx
======

Core **JraphX** functionality and version information.

.. currentmodule:: jraphx

Core Classes
------------

.. autosummary::
   :nosignatures:

   Data
   Batch

Version Information
-------------------

.. automodule:: jraphx
    :members: __version__
    :undoc-members:

Quick Start
-----------

**JraphX** provides a simple, JAX-based interface for graph neural networks:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data
    from jraphx.nn.conv import GCNConv
    from flax import nnx

    # Create a graph
    x = jnp.ones((10, 16))
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])
    data = Data(x=x, edge_index=edge_index)

    # Create and use a GNN layer
    layer = GCNConv(16, 32, rngs=nnx.Rngs(42))
    output = layer(data.x, data.edge_index)

    print(f"Output shape: {output.shape}")

Submodules
----------

.. autosummary::
   :nosignatures:

   data
   nn
   utils

JAX Integration
---------------

**JraphX** is designed from the ground up for JAX:

- All operations are **pure functions**
- Full support for **@jax.jit compilation**
- Compatible with **jax.vmap** and **nnx.vmap** for batching
- Integrates with **jax.grad** for automatic differentiation
- Works seamlessly with **Optax** optimizers
