Test Coverage
=============

This document records what the JraphX test suite covers, and which
:obj:`torch_geometric` test areas are not covered because the underlying feature
does not exist or is out of scope.

Current Coverage
----------------

The suite contains 1215 tests. Counts below are collected test cases, so
parametrized tests contribute one entry per parameter set.

Core Data Structures
~~~~~~~~~~~~~~~~~~~~

- ``tests/data/test_data.py`` -- ``Data`` construction, attribute access, node and
  edge counts, pytree round-trips (30)
- ``tests/data/test_batch.py`` -- ``Batch`` collation, node-index offsetting,
  unbatching, subclass configuration, pytree behavior under ``jax.jit`` (47)

Convolution Layers
~~~~~~~~~~~~~~~~~~

- ``tests/nn/conv/test_message_passing.py`` -- the ``propagate`` contract: both
  flow directions, bipartite and homogeneous input, single evaluation of
  ``message()``, and the opt-in ``message_and_aggregate`` hook (15)
- ``tests/nn/conv/test_gcn_conv.py`` -- symmetric normalization, weighted degree,
  ``improved``, self-loop handling, the explicit ``precompute_norm`` cache (28)
- ``tests/nn/conv/test_gat_conv.py`` -- per-head attention normalization, edge
  features, multi-head concat and averaging (20)
- ``tests/nn/conv/test_gatv2_conv.py`` -- GATv2 attention, including the per-head
  softmax property (15)
- ``tests/nn/conv/test_sage_conv.py`` -- aggregations, root weight, bipartite
  input with and without an explicit ``size`` (20)
- ``tests/nn/conv/test_gin_conv.py`` -- ``eps`` and ``train_eps``, bipartite input (11)
- ``tests/nn/conv/test_edge_conv.py`` -- ``EdgeConv`` and ``DynamicEdgeConv``,
  including single evaluation of a stateful inner network (19)
- ``tests/nn/conv/test_transformer_conv.py`` -- scaled dot-product attention,
  edge features on both keys and values, ``beta`` and root weight (18)

Utility Functions
~~~~~~~~~~~~~~~~~

- ``tests/utils/test_scatter.py`` -- ``scatter_add``/``mean``/``max``/``min``/
  ``std``/``logsumexp``, dtype preservation, empty segments (25)
- ``tests/utils/test_scatter_softmax.py`` -- ``scatter_softmax``,
  ``scatter_log_softmax``, ``masked_scatter_softmax``, fully masked groups (15)
- ``tests/utils/test_coalesce.py`` -- reduction modes, sortedness, large node
  counts (13)
- ``tests/utils/test_loop.py`` -- self-loop addition and removal, ``fill_value`` (18)
- ``tests/utils/test_degree.py`` -- ``degree``, ``in_degree``, ``out_degree`` (9)
- ``tests/utils/test_convert.py`` -- ``to_undirected``, ``to_dense_adj``,
  ``to_edge_index`` (13)

Models
~~~~~~

- ``tests/nn/models/test_basic_gnn.py`` -- the prebuilt ``GCN``, ``GAT``,
  ``GraphSAGE`` and ``GIN`` stacks across layer counts, normalizations, jumping
  knowledge modes and dropout (776)
- ``tests/nn/models/test_mlp.py`` -- feature lists, normalization, ``plain_last`` (23)
- ``tests/nn/models/test_jumping_knowledge.py`` -- ``cat``, ``max`` and ``lstm``
  modes (9)

Pooling Operations
~~~~~~~~~~~~~~~~~~

- ``tests/nn/pool/test_glob.py`` -- global add/mean/max/min/sort/softmax pooling,
  empty graphs, node features of arbitrary rank (24)
- ``tests/nn/pool/test_topk_pool.py`` -- ``TopKPooling`` and ``SAGPooling``:
  ``ratio`` typing, ``min_score``, the score gate and edge relabeling (26)

Normalization Layers
~~~~~~~~~~~~~~~~~~~~

- ``tests/nn/norm/test_layer_norm.py`` -- node and graph modes, affine
  parameters (19)
- ``tests/nn/norm/test_graph_norm.py`` -- per-feature per-graph statistics and the
  learnable mean shift (10)
- ``tests/nn/norm/test_batch_norm.py`` -- running statistics, training and
  evaluation modes (9)

Packaging
~~~~~~~~~

- ``tests/test_version.py`` -- keeps ``__version__`` and the newest changelog
  heading in lockstep (3)

Not Covered - Feature Not Implemented
-------------------------------------

These :obj:`torch_geometric` test areas have no JraphX counterpart because the
feature itself is absent. See :doc:`missing_features` for the full list.

Convolution Layers
~~~~~~~~~~~~~~~~~~

- ``test_appnp.py``, ``test_cheb_conv.py``, ``test_arma_conv.py``,
  ``test_graph_conv.py``, ``test_nn_conv.py``, ``test_spline_conv.py``,
  ``test_pna_conv.py``, ``test_film_conv.py``

Heterogeneous Graphs
~~~~~~~~~~~~~~~~~~~~

- ``test_hetero_conv.py``, ``test_hgt_conv.py``, ``test_han_conv.py`` -- there is
  no heterogeneous graph support

Aggregation Modules
~~~~~~~~~~~~~~~~~~~

- Tests for the ``Aggregation`` class hierarchy. JraphX exposes aggregation as
  scatter functions and pooling operations rather than as composable modules, so
  ``MultiAggregation``, ``DegreeScalerAggregation``, ``Set2Set`` and friends have
  no counterpart.

Hierarchical Pooling
~~~~~~~~~~~~~~~~~~~~

- ``test_asap.py``, ``test_diff_pool.py``, ``test_edge_pool.py``.
  ``TopKPooling`` and ``SAGPooling`` *are* implemented and tested above.

Advanced Models
~~~~~~~~~~~~~~~

- ``test_autoencoder.py``, ``test_deep_graph_infomax.py``, ``test_node2vec.py``,
  ``test_metapath2vec.py``

Not Covered - PyTorch-Specific
------------------------------

These rely on PyTorch functionality with no JAX equivalent:

- ``torch.jit.script()`` compilation and TorchScript behavior. The JAX analog
  is :obj:`jax.jit`, which the suite exercises directly where a layer supports it.
- ``test_fused_gat_conv.py``, the ``tests/nn/conv/cugraph/`` tree, and GPU memory
  management -- CUDA kernel specialisations.
- ``test_datamodule.py`` -- PyTorch Lightning integration.
- ``test_sparse.py`` and ``torch_sparse`` integration -- JraphX represents graphs
  with dense ``edge_index`` arrays.

Not Covered - Out of Scope
--------------------------

- ``tests/explain/`` -- GNN explainability is a specialized domain outside core
  GNN functionality.
- ``tests/visualization/`` -- handled by specialized libraries.
- ``tests/nn/nlp/`` -- sentence transformers and language model integration.
- Point cloud convolutions (``test_point_conv.py``, ``test_ppf_conv.py``) and
  molecular property prediction models.
- ``test_correct_and_smooth.py`` and label propagation.
- ``tests/data/test_database.py`` and ``tests/data/test_remote_backend_utils.py``
  -- database and remote storage backends.

Conventions for New Tests
-------------------------

Imports
~~~~~~~

.. code-block:: python

   # PyTorch Geometric
   import torch
   from torch_geometric.nn import GCNConv
   from torch_geometric.data import Data, Batch

   # JraphX
   import jax.numpy as jnp
   from jraphx.nn.conv import GCNConv
   from jraphx.data import Data, Batch

Array Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # PyTorch
   x = torch.randn(4, 16)
   assert torch.allclose(x, x)

   # JAX
   x = random.normal(random.key(0), (4, 16))
   assert jnp.allclose(x, x)

Module Construction
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # PyTorch Geometric
   conv = GCNConv(16, 32)

   # JraphX
   conv = GCNConv(16, 32, rngs=nnx.Rngs(0))

Guidelines
~~~~~~~~~~

- Pin numerics, not just shapes. A test for a numerical fix that asserts only
  ``out.shape`` passes against the bug it was written for.
- Prefer small deterministic literal inputs over PRNG draws when asserting exact
  values, so the constants stay stable across JAX versions and platforms.
- Do not wrap assertions in ``try``/``except``; let failures propagate.
- Run the suite with ``-W error::DeprecationWarning``; it is currently clean.

Keep this document in step with the suite when adding tests or features.
