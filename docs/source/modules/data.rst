Working with Graph Data
=======================

This guide covers how to work with graph data structures in JraphX. JraphX uses ``flax.struct.dataclass`` for its data structures, making them fully compatible with JAX transformations like ``jit``, ``vmap``, ``grad``, and ``pmap``.

.. contents:: Contents
    :local:

Core Data Classes
-----------------

.. currentmodule:: jraphx.data

.. autosummary::
   :nosignatures:

   Data
   Batch

The Data Class
--------------

The ``Data`` class is the fundamental data structure for representing graphs in JraphX. It is immutable and registered as a PyTree for efficient JAX operations.

.. code-block:: python

    from jraphx import Data
    import jax.numpy as jnp

    # Create a graph
    data = Data(
        x=jnp.array([[1.0], [2.0], [3.0]]),  # Node features [num_nodes, num_features]
        edge_index=jnp.array([[0, 1, 2], [1, 2, 0]]),  # Edge indices [2, num_edges]
        y=jnp.array([0])  # Graph label
    )

    # Access properties
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_node_features}")

Key Characteristics
^^^^^^^^^^^^^^^^^^^

**Immutability**

Data objects are immutable to ensure functional purity:

.. code-block:: python

    # Cannot modify attributes directly
    # data.x = new_x  # This will raise an error

    # Use replace() to create a modified copy
    new_data = data.replace(x=data.x * 2)

**JAX Compatibility**

Data objects work seamlessly with JAX transformations:

.. code-block:: python

    import jax

    # JIT compilation
    @jax.jit
    def process_graph(data):
        return data.x.sum()

    # Vectorization
    batched_process = jax.vmap(process_graph)

    # Device placement
    data_on_gpu = jax.device_put(data, jax.devices()[0])

**PyTree Operations**

As registered PyTrees, Data objects support tree operations:

.. code-block:: python

    # Apply function to all arrays
    data_float32 = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x is not None else None,
        data
    )

Graph Batching
--------------

The ``Batch`` class efficiently combines multiple graphs into a single disconnected graph:

.. code-block:: python

    from jraphx import Data, Batch

    # Create individual graphs
    graph1 = Data(
        x=jnp.array([[1.0], [2.0]]),
        edge_index=jnp.array([[0], [1]]),
        y=jnp.array([0])
    )

    graph2 = Data(
        x=jnp.array([[3.0], [4.0], [5.0]]),
        edge_index=jnp.array([[0, 1], [1, 2]]),
        y=jnp.array([1])
    )

    # Batch them together
    batch = Batch.from_data_list([graph1, graph2])

    print(f"Batched nodes: {batch.num_nodes}")  # 5 total
    print(f"Batched edges: {batch.num_edges}")  # 3 total
    print(f"Batch vector: {batch.batch}")  # [0, 0, 1, 1, 1]

The batch vector indicates which graph each node belongs to, enabling proper pooling operations:

.. code-block:: python

    from jraphx.nn.pool import global_mean_pool

    # Process batched graphs
    node_embeddings = model(batch.x, batch.edge_index)

    # Pool to graph-level representations
    graph_embeddings = global_mean_pool(node_embeddings, batch.batch)
    print(f"Graph embeddings shape: {graph_embeddings.shape}")  # [2, hidden_dim]

Extending the Data and Batch Classes
------------------------------------

For domain-specific attributes, subclass both the base ``Data`` and ``Batch`` classes:

Simple Data Extension
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from typing import Optional
    from flax.struct import dataclass
    import jraphx

    @dataclass
    class CitationData(Data):
        """Data class for citation networks with train/val/test splits."""
        train_mask: jnp.ndarray | None = None
        val_mask: jnp.ndarray | None = None
        test_mask: jnp.ndarray | None = None

    # Use the extended class
    citation_data = CitationData(
        x=node_features,
        edge_index=edges,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

Extending Both Data and Batch Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For custom Data subclasses with additional fields, create a corresponding Batch subclass and specify batching behavior using class attributes:

.. code-block:: python

    from flax.struct import dataclass
    from typing import Optional
    import jraphx

    @dataclass
    class FaceData(Data):
        """Data class for 3D mesh graphs with face connectivity."""
        face: jnp.ndarray | None = None       # Face connectivity [3, num_faces]
        pos: jnp.ndarray | None = None        # 3D node positions
        normal: jnp.ndarray | None = None     # Face normals
        face_color: jnp.ndarray | None = None # Face colors

    @dataclass
    class FaceBatch(jraphx.Batch):
        """Batch class for 3D mesh graphs."""
        face: jnp.ndarray | None = None
        pos: jnp.ndarray | None = None
        normal: jnp.ndarray | None = None
        face_color: jnp.ndarray | None = None

        # Configure batching behavior as class attributes
        NODE_INDEX_FIELDS = {'face'}
        ELEMENT_LEVEL_FIELDS = {'normal', 'face_color', 'pos'}
        _DATA_CLASS = FaceData  # Link for unbatching

        def __repr__(self) -> str:
            """Use the nice shape-based representation from parent class."""
            return jraphx.Batch.__repr__(self)

    # Create mesh graphs
    mesh1 = FaceData(
        x=jnp.ones((4, 3)),  # 4 vertices
        face=jnp.array([[0, 1, 2], [1, 2, 3]]).T,  # 2 triangular faces
        normal=jnp.array([[0., 0., 1.], [0., 1., 0.]]),  # Face normals
        face_color=jnp.array([[1., 0., 0.], [0., 1., 0.]])  # Red and green
    )

    mesh2 = FaceData(
        x=jnp.ones((3, 3)),  # 3 vertices
        face=jnp.array([[0, 1, 2]]).T,  # 1 triangular face
        normal=jnp.array([[1., 0., 0.]]),  # Face normal
        face_color=jnp.array([[0., 0., 1.]])  # Blue
    )

    # Batch them together
    batch = FaceBatch.from_data_list([mesh1, mesh2])

    # Unbatch
    meshes = batch.to_data_list()  # Returns list of FaceData objects

The batching system provides three configuration options:

- **NODE_INDEX_FIELDS**: Fields containing node indices that need adjustment during batching (like ``edge_index`` or ``face``)
- **ELEMENT_LEVEL_FIELDS**: Fields that are element-level features aligned with a node index field (concatenated during batching)
- **GRAPH_LEVEL_FIELDS**: Fields that are per-graph attributes (stacked, not concatenated)

Example: Molecular Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @dataclass
    class MolecularData(Data):
        """Data class for molecular graphs."""
        bond_index: jnp.ndarray | None = None  # Bond connectivity
        bond_type: jnp.ndarray | None = None   # Bond type features
        atom_charge: jnp.ndarray | None = None # Node-level charges
        mol_weight: float | None = None        # Graph-level property

    @dataclass
    class MolecularBatch(jraphx.Batch):
        """Batch class for molecular graphs."""
        bond_index: jnp.ndarray | None = None
        bond_type: jnp.ndarray | None = None
        atom_charge: jnp.ndarray | None = None
        mol_weight: jnp.ndarray | None = None

        # Configure batching behavior as class attributes
        NODE_INDEX_FIELDS = {'bond_index'}
        ELEMENT_LEVEL_FIELDS = {'bond_type', 'atom_charge'}
        GRAPH_LEVEL_FIELDS = {'mol_weight'}  # Per-molecule property
        _DATA_CLASS = MolecularData  # Link for unbatching

        def __repr__(self) -> str:
            """Use the nice shape-based representation from parent class."""
            return jraphx.Batch.__repr__(self)

    # Create molecules
    mol1 = MolecularData(
        x=jnp.array([[6.], [1.], [1.]]),  # C, H, H
        edge_index=jnp.array([[0, 0], [1, 2]]),
        bond_index=jnp.array([[0, 0], [1, 2]]),
        bond_type=jnp.array([[1.], [1.]]),  # Single bonds
        atom_charge=jnp.array([0., 0., 0.]),
        mol_weight=16.04
    )

    mol2 = MolecularData(
        x=jnp.array([[8.], [1.]]),  # O, H
        edge_index=jnp.array([[0], [1]]),
        bond_index=jnp.array([[0], [1]]),
        bond_type=jnp.array([[1.]]),  # Single bond
        atom_charge=jnp.array([-0.5, 0.5]),
        mol_weight=17.01
    )

    # Batch molecules
    batch = MolecularBatch.from_data_list([mol1, mol2])

    # Access graph-level properties
    print(f"Molecular weights: {batch.mol_weight}")  # [16.04, 17.01]

.. _3d-mesh-graphs:

Example: 3D Mesh Graphs
^^^^^^^^^^^^^^^^^^^^^^^

Using the FaceData and FaceBatch classes defined above, here's how to create and batch 3D mesh graphs:

.. code-block:: python

    # Create a simple triangle mesh (tetrahedron)
    mesh_graph = FaceData(
        x=jnp.ones((4, 1)),  # 4 vertices with dummy features
        edge_index=jnp.array([[0, 1, 2, 0, 1, 2], [1, 2, 0, 3, 3, 3]]),  # Edges
        face=jnp.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 3]]),  # 4 triangular faces
        pos=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]]),  # 3D positions
        normal=jnp.array([[0., 0., 1.], [0., 1., 0.],
                          [1., 0., 0.], [0.5, 0.5, 0.5]]),  # Face normals
        face_color=jnp.array([[1., 0., 0.], [0., 1., 0.],
                              [0., 0., 1.], [1., 1., 0.]])  # Face colors
    )

    # Batch multiple meshes (see Graph Batching section for details)
    batch = FaceBatch.from_data_list([mesh_graph, mesh_graph, mesh_graph])

    # Unbatch back to individual FaceData objects
    meshes = batch.to_data_list()


Working with PyTorch Geometric
-------------------------------

When converting from PyTorch Geometric datasets, create a custom Data class:

.. code-block:: python

    @dataclass
    class PyGData(Data):
        """Data class compatible with PyTorch Geometric datasets."""
        train_mask: jnp.ndarray | None = None
        val_mask: jnp.ndarray | None = None
        test_mask: jnp.ndarray | None = None
        edge_attr: jnp.ndarray | None = None

    def from_pyg(pyg_data):
        """Convert PyTorch Geometric data to JraphX format."""
        return PyGData(
            x=jnp.array(pyg_data.x.numpy()),
            edge_index=jnp.array(pyg_data.edge_index.numpy()),
            y=jnp.array(pyg_data.y.numpy()),
            train_mask=jnp.array(pyg_data.train_mask.numpy()),
            val_mask=jnp.array(pyg_data.val_mask.numpy()),
            test_mask=jnp.array(pyg_data.test_mask.numpy()),
            edge_attr=jnp.array(pyg_data.edge_attr.numpy())
                if hasattr(pyg_data, 'edge_attr') else None
        )

Common Patterns
---------------

Device Management
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Move entire graph to GPU
    device = jax.devices('gpu')[0]
    data_gpu = jax.device_put(data, device)

    # Check device placement
    print(f"Data on device: {data_gpu.x.device()}")

Preprocessing
^^^^^^^^^^^^^

.. code-block:: python

    def normalize_features(data: Data) -> Data:
        """Normalize node features to zero mean and unit variance."""
        x = data.x  # [num_nodes, num_node_features]
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        x_normalized = (x - mean) / (std + 1e-6)
        return data.replace(x=x_normalized)

    # Apply normalization
    data_normalized = normalize_features(data)

Data Augmentation
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from functools import partial

    @partial(jax.jit, donate_argnums=0)
    def add_noise(data: Data, rng: jax.Array, noise_level: float = 0.1) -> Data:
        """Add Gaussian noise to node features."""
        noise = random.normal(rng, data.x.shape) * noise_level
        return data.replace(x=data.x + noise)

    @partial(jax.jit, donate_argnums=0)
    def drop_edges(data: Data, rng: jax.Array, drop_rate: float = 0.1) -> Data:
        """Randomly drop edges for augmentation."""
        num_edges = data.edge_index.shape[1]
        mask = random.bernoulli(rng, shape=(num_edges,), p=1-drop_rate)
        new_edge_index = data.edge_index[:, mask]
        return data.replace(edge_index=new_edge_index)

Performance Considerations
--------------------------

Memory Efficiency
^^^^^^^^^^^^^^^^^

- **Immutability**: Creates new objects for modifications, but JAX's XLA compiler optimizes this. Consider using ``donate_argnums``/``donate_argnames`` with ``jax.jit``/``nnx.jit`` and related functions.
- **PyTree operations**: Very efficient for batch operations
- **Subclassing**: No overhead - only stores defined attributes

JIT Compilation
^^^^^^^^^^^^^^^

.. code-block:: python

    @jax.jit
    def efficient_forward(data: Data, params):
        # All operations on Data work with JIT
        x = data.x
        edge_index = data.edge_index
        return model.apply(params, x, edge_index)

Large Graphs
^^^^^^^^^^^^

For very large graphs that don't fit in memory:

.. code-block:: python

    def process_large_graph_in_chunks(data: Data, chunk_size: int = 1000):
        """Process large graphs in chunks using scan."""
        num_nodes = data.num_nodes
        num_chunks = (num_nodes + chunk_size - 1) // chunk_size

        def process_chunk(carry, chunk_idx):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_nodes)
            chunk_x = data.x[start:end]
            # Process chunk...
            return carry, chunk_output

        _, outputs = jax.lax.scan(process_chunk, None, jnp.arange(num_chunks))
        return outputs

Best Practices
--------------

1. **Always subclass Data** for domain-specific attributes rather than trying to modify instances
2. **Use Optional types** for attributes that may not always be present
3. **Leverage immutability** for reproducible and debuggable code
4. **Use replace()** method for creating modified instances
5. **Take advantage of PyTree operations** for efficient batch processing
6. **Prefer JAX arrays** over Python lists or NumPy arrays for all tensor data

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**AttributeError when setting attributes**

.. code-block:: python

    # Wrong
    data.custom_attr = value  # Raises AttributeError

    # Right - subclass Data
    @dataclass
    class MyData(Data):
        custom_attr: jnp.ndarray | None = None

    data = MyData(x=x, edge_index=edges, custom_attr=value)

**Type errors with JAX transforms**

Ensure all attributes are JAX-compatible types or mark non-JAX attributes:

.. code-block:: python

    from flax import struct

    @dataclass
    class DataWithMetadata(Data):
        # JAX array - will be traced
        features: jnp.ndarray | None = None

        # Non-JAX metadata - won't be traced
        name: str = struct.field(pytree_node=False, default="")

JAX-Specific Features
---------------------

All **JraphX** data structures are designed for JAX:

- **Immutable**: Data objects are immutable, operations return new instances
- **JIT-Compatible**: All operations work with :obj:`@jax.jit`
- **Pure Functions**: No side effects, functional programming friendly
- **Device Agnostic**: Works on CPU, GPU, and TPU seamlessly

See Also
--------

- :doc:`/tutorial/examples` - Tutorials demonstrating Data usage
- :doc:`/get_started/introduction` - Introduction to JraphX concepts
- `Flax Struct Documentation <https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html>`_ - Details on flax.struct.dataclass
