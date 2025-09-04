jraphx.utils
============

.. currentmodule:: jraphx.utils

Utility functions and operations for graph processing and manipulation.

Scatter Operations
------------------

.. currentmodule:: jraphx.utils.scatter

The scatter module provides efficient implementations of scatter operations for aggregating node features.

scatter_sum
~~~~~~~~~~~

.. autofunction:: scatter_sum

   Scatter sum operation for aggregating values by index.

   **Example:**

   .. code-block:: python

      from jraphx.utils.scatter import scatter_sum
      import jax.numpy as jnp

      src = jnp.array([1.0, 2.0, 3.0, 4.0])
      index = jnp.array([0, 0, 1, 1])

      # Sum values by index
      out = scatter_sum(src, index, dim_size=2)
      # Result: [3.0, 7.0]

scatter_mean
~~~~~~~~~~~~

.. autofunction:: scatter_mean

   Scatter mean operation for averaging values by index.

   **Example:**

   .. code-block:: python

      from jraphx.utils.scatter import scatter_mean
      import jax.numpy as jnp

      src = jnp.array([1.0, 2.0, 3.0, 4.0])
      index = jnp.array([0, 0, 1, 1])

      # Average values by index
      out = scatter_mean(src, index, dim_size=2)
      # Result: [1.5, 3.5]

scatter_max
~~~~~~~~~~~

.. autofunction:: scatter_max

   Scatter max operation for finding maximum values by index.

   **Example:**

   .. code-block:: python

      from jraphx.utils.scatter import scatter_max
      import jax.numpy as jnp

      src = jnp.array([1.0, 3.0, 2.0, 4.0])
      index = jnp.array([0, 0, 1, 1])

      # Find max values by index
      out = scatter_max(src, index, dim_size=2)
      # Result: [3.0, 4.0]

scatter_min
~~~~~~~~~~~

.. autofunction:: scatter_min

   Scatter min operation for finding minimum values by index.

scatter_std
~~~~~~~~~~~

.. autofunction:: scatter_std

   Scatter standard deviation operation.

scatter
~~~~~~~

.. autofunction:: scatter

   Generic scatter operation with configurable reduction.

   **Parameters:**

   - **src**: Source tensor to scatter
   - **index**: Index tensor for scattering
   - **dim**: Dimension to scatter along
   - **dim_size**: Size of the output dimension
   - **reduce**: Reduction operation ('sum', 'mean', 'max', 'min', 'mul')

Graph Utilities
---------------

.. currentmodule:: jraphx.utils

degree
~~~~~~

.. autofunction:: degree

   Compute the degree of each node in a graph.

   **Example:**

   .. code-block:: python

      from jraphx.utils import degree
      import jax.numpy as jnp

      edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

      # Compute in-degree and out-degree
      in_deg = degree(edge_index[1], num_nodes=3)
      out_deg = degree(edge_index[0], num_nodes=3)

to_undirected
~~~~~~~~~~~~~

.. autofunction:: to_undirected

   Convert a directed graph to undirected by adding reverse edges.

   **Example:**

   .. code-block:: python

      from jraphx.utils import to_undirected
      import jax.numpy as jnp

      edge_index = jnp.array([[0, 1], [1, 2]])
      edge_attr = jnp.array([[1.0], [2.0]])

      # Convert to undirected
      edge_index_undirected, edge_attr_undirected = to_undirected(
          edge_index, edge_attr
      )

add_self_loops
~~~~~~~~~~~~~~

.. autofunction:: add_self_loops

   Add self-loop edges to a graph.

   **Example:**

   .. code-block:: python

      from jraphx.utils import add_self_loops
      import jax.numpy as jnp

      edge_index = jnp.array([[0, 1], [1, 2]])

      # Add self-loops
      edge_index_with_loops, edge_attr = add_self_loops(
          edge_index, num_nodes=3
      )

remove_self_loops
~~~~~~~~~~~~~~~~~

.. autofunction:: remove_self_loops

   Remove self-loop edges from a graph.

contains_self_loops
~~~~~~~~~~~~~~~~~~~

.. autofunction:: contains_self_loops

   Check if a graph contains self-loop edges.

is_undirected
~~~~~~~~~~~~~

.. autofunction:: is_undirected

   Check if a graph is undirected.

coalesce
~~~~~~~~

.. autofunction:: coalesce

   Remove duplicate edges and optionally sum their attributes.

   **Example:**

   .. code-block:: python

      from jraphx.utils import coalesce
      import jax.numpy as jnp

      # Graph with duplicate edges
      edge_index = jnp.array([[0, 0, 1], [1, 1, 2]])
      edge_attr = jnp.array([[1.0], [2.0], [3.0]])

      # Remove duplicates and sum attributes
      edge_index_clean, edge_attr_clean = coalesce(
          edge_index, edge_attr, reduce='sum'
      )

Conversion Utilities
--------------------

to_dense_adj
~~~~~~~~~~~~

.. autofunction:: to_dense_adj

   Convert edge indices to a dense adjacency matrix.

   **Example:**

   .. code-block:: python

      from jraphx.utils import to_dense_adj
      import jax.numpy as jnp

      edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

      # Convert to dense adjacency matrix
      adj = to_dense_adj(edge_index, num_nodes=3)

to_edge_index
~~~~~~~~~~~~~

.. autofunction:: to_edge_index

   Convert adjacency representation to edge index format.
