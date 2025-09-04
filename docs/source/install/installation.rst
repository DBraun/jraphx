Installation
============

**JraphX** is a Graph Neural Network library for JAX/Flax NNX, available for Python 3.11+.

Quick Start
-----------

**Option 1: Manual Installation**

Follow the `official JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`__ to install JAX for your platform (CPU, GPU, or TPU).

**Option 2: JAX AI Stack**

An easy way to get started is with the `JAX AI Stack repository <https://github.com/jax-ml/jax-ai-stack>`__, which includes JAX, Flax, Optax, and other ML libraries. However, we don't recommend this approach yet, since we need Flax 0.11.2, and jax-ai-stack 2025.9.3 is pinned to exactly 0.11.1.

.. code-block:: bash

   pip install jax-ai-stack

Install JraphX
--------------

Clone the JraphX repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/DBraun/jraphx.git
   cd jraphx
   pip install -e .

Verification
------------

To verify your installation is working correctly:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from flax import nnx
   import jraphx

   print(f"JAX version: {jax.__version__}")
   print(f"JAX backend: {jax.default_backend()}")
   print(f"JraphX version: {jraphx.__version__}")

   # Test basic functionality
   from jraphx.data import Data
   from jraphx.nn.conv import GCNConv

   # Create a simple graph
   data = Data(
       x=jnp.ones((3, 4)),
       edge_index=jnp.array([[0, 1, 2], [1, 2, 0]])
   )

   # Create and use a GNN layer
   layer = GCNConv(4, 8, rngs=nnx.Rngs(42))
   output = layer(data.x, data.edge_index)
   print(f"Successfully processed graph: {output.shape}")

Troubleshooting
---------------

**Import Error:** If you get "No module named 'jraphx'", make sure you installed with `pip install -e .` from the jraphx directory.

**JAX Issues:** Refer to the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`__ for platform-specific troubleshooting.

For other issues, please create an issue on the `JraphX GitHub repository <https://github.com/DBraun/jraphx/issues>`__.
