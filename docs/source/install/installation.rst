Installation
============

**JraphX** is a Graph Neural Network library for JAX/Flax NNX, available for Python 3.11+.

Quick Start
-----------

**Option 1: Install from PyPI (Recommended)**

Install JraphX directly from PyPI:

.. code-block:: bash

   pip install jraphx

This will automatically install the required dependencies: JAX, Flax, and NumPy.

**Option 2: JAX AI Stack + JraphX**

The `JAX AI Stack <https://github.com/jax-ml/jax-ai-stack>`__ provides a curated collection of JAX, Flax, Optax, and other ML libraries. After installing it, you can add JraphX:

.. code-block:: bash

   pip install jax-ai-stack
   pip install jraphx

.. note::
   JraphX requires Flax 0.12.0 or higher. JAX AI Stack 2025.10.28 and later includes Flax 0.12.0, so this installation method works.

Development Installation
------------------------

For development or to get the latest features, clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/DBraun/jraphx.git
   cd jraphx
   pip install -e .

You can also use this approach after installing JAX AI Stack:

.. code-block:: bash

   pip install jax-ai-stack
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
