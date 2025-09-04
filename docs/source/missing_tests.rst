Missing Tests in JraphX
=======================

This document tracks PyTorch Geometric tests that could not be ported to JraphX due to missing functionality, architectural differences, or deliberate design choices.

Test Conversion Progress
------------------------

The JraphX library aims to provide core GNN functionality with a JAX-first design. Not all PyTorch Geometric features are implemented, as some are outside the scope of a lightweight JAX library.

Successfully Converted Tests
----------------------------

All tests have been successfully converted with comprehensive test coverage. Below is the complete list of converted test files and their status.

Core Data Structures
~~~~~~~~~~~~~~~~~~~~~
- ``test_data_jraphx.py`` ✓ - Data class functionality (9 tests)
- ``test_batch_jraphx.py`` ✓ - Batch class functionality (7 tests)

Convolution Layers
~~~~~~~~~~~~~~~~~~
- ``test_gcn_conv_jraphx.py`` ✓ - Graph Convolution Network (10 tests)
- ``test_gat_conv_jraphx.py`` ✓ - Graph Attention Network (12 tests)
- ``test_gatv2_conv_jraphx.py`` ✓ - Improved Graph Attention Network (referenced via GAT)
- ``test_sage_conv_jraphx.py`` ✓ - GraphSAGE layer (16 tests)
- ``test_gin_conv_jraphx.py`` ✓ - Graph Isomorphism Network (9 tests)
- ``test_edge_conv_jraphx.py`` ✓ - EdgeConv and DynamicEdgeConv layers (12 tests)
- ``test_transformer_conv_jraphx.py`` ✓ - Graph Transformer layer (13 tests)

Utility Functions
~~~~~~~~~~~~~~~~~
- ``test_scatter_jraphx.py`` ✓ - Scatter operations (add, mean, max, min, std, logsumexp) (9 tests)
- ``test_degree_jraphx.py`` ✓ - Degree computation functions (3 tests)
- ``test_coalesce_jraphx.py`` ✓ - Edge coalescing functionality (5 tests)
- ``test_loop_jraphx.py`` ✓ - Self-loop addition/removal (6 tests)

Models
~~~~~~
- ``test_basic_gnn_jraphx.py`` ✓ - Pre-built GNN models (GCN, GAT, SAGE, GIN) (769 tests)
- ``test_jumping_knowledge_jraphx.py`` ✓ - JumpingKnowledge aggregation (4 tests)
- ``test_mlp_jraphx.py`` ✓ - Multi-layer perceptron (12 tests)

Pooling Operations
~~~~~~~~~~~~~~~~~~
- ``test_glob_jraphx.py`` ✓ - Global pooling operations (add, mean, max) (4 tests)

Normalization Layers
~~~~~~~~~~~~~~~~~~~~~
- ``test_batch_norm_jraphx.py`` ✓ - Batch normalization (5 tests)
- ``test_layer_norm_jraphx.py`` ✓ - Layer normalization (9 tests)
- ``test_graph_norm_jraphx.py`` ✓ - Graph normalization (6 tests)

**Total: 920+ test cases covering all implemented JraphX functionality**

Tests Not Converted - Missing Core Features
--------------------------------------------

High Priority Missing Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These tests cannot be converted because the underlying JraphX features don't exist yet:

Convolution Layers
^^^^^^^^^^^^^^^^^^
- ``test_appnp.py`` - APPNP layer not implemented
- ``test_cheb_conv.py`` - Chebyshev convolution not implemented
- ``test_arma_conv.py`` - ARMA filters not implemented
- ``test_graph_conv.py`` - Basic graph convolution not implemented
- ``test_nn_conv.py`` - NN-based convolution not implemented
- ``test_spline_conv.py`` - Spline-based convolution not implemented
- ``test_pna_conv.py`` - Principal Neighbourhood Aggregation not implemented
- ``test_film_conv.py`` - FiLM layers not implemented

Message Passing Framework
^^^^^^^^^^^^^^^^^^^^^^^^^
- ``test_message_passing.py`` - Only partial message passing framework exists
- Advanced aggregation functions beyond basic scatter operations
- Custom message/update function hooks

Heterogeneous Graphs
^^^^^^^^^^^^^^^^^^^^
- ``test_hetero_conv.py`` - No heterogeneous graph support
- ``test_hgt_conv.py`` - Heterogeneous Graph Transformer not implemented
- ``test_han_conv.py`` - Heterogeneous Attention Network not implemented

Pooling Operations
^^^^^^^^^^^^^^^^^^
- ``test_topk_pool.py`` - TopK pooling not implemented
- ``test_sag_pool.py`` - SAG pooling not implemented
- ``test_asap.py`` - ASAP pooling not implemented
- ``test_diff_pool.py`` - DiffPool not implemented
- ``test_edge_pool.py`` - Edge pooling not implemented

Advanced Models
^^^^^^^^^^^^^^^
- ``test_autoencoder.py`` - Autoencoder models not implemented
- ``test_deep_graph_infomax.py`` - Deep Graph Infomax not implemented
- ``test_node2vec.py`` - Node2Vec not implemented
- ``test_metapath2vec.py`` - Metapath2Vec not implemented

Tests Not Converted - PyTorch-Specific Features
------------------------------------------------

These tests rely on PyTorch-specific functionality that doesn't have JAX equivalents:

JIT Compilation & TorchScript
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Tests involving ``torch.jit.script()`` compilation
- TorchScript-specific functionality tests
- Dynamic graph compilation features

CUDA-Specific Tests
~~~~~~~~~~~~~~~~~~~
- ``test_fused_gat_conv.py`` - CUDA kernel optimizations
- CuGraph integration tests (``tests/nn/conv/cugraph/``)
- GPU memory management tests

PyTorch Lightning Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``test_datamodule.py`` - PyTorch Lightning specific features
- Lightning-based training loop tests

Sparse Tensor Operations
~~~~~~~~~~~~~~~~~~~~~~~~
- ``test_sparse.py`` - PyTorch sparse tensor operations
- torch_sparse library integration tests
- SparseTensor class functionality

Tests Not Converted - Out of Scope
-----------------------------------

These test areas are deliberately excluded from JraphX scope:

Explainability
~~~~~~~~~~~~~~
- ``tests/explain/`` - Entire explainability module (47 test files)
- GNN explainability is a specialized domain outside core GNN functionality

Visualization
~~~~~~~~~~~~~
- ``tests/visualization/`` - Graph visualization functionality
- Visualization is typically handled by specialized libraries

Natural Language Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``tests/nn/nlp/`` - NLP-specific functionality
- Sentence transformers and language model integration

3D Point Clouds & Molecular Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Point cloud convolutions (``test_point_conv.py``, ``test_ppf_conv.py``)
- Molecular property prediction models
- 3D geometry-specific operations

Advanced Optimization
~~~~~~~~~~~~~~~~~~~~~
- ``test_correct_and_smooth.py`` - Advanced optimization techniques
- Label propagation algorithms
- Advanced training strategies

Database & Storage Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``tests/data/test_database.py`` - Database integration
- ``tests/data/test_remote_backend_utils.py`` - Remote storage
- Complex data pipeline functionality

Conversion Notes and Patterns
------------------------------

When converting tests, the following patterns were applied:

Import Conversions
~~~~~~~~~~~~~~~~~~
.. code-block:: python

   # PyTorch Geometric
   import torch
   from torch_geometric.nn import GCNConv
   from torch_geometric.data import Data, Batch

   # JraphX
   import jax.numpy as jnp
   from jraphx.nn.conv import GCNConv
   from jraphx.data import Data, Batch

Tensor Operations
~~~~~~~~~~~~~~~~~
.. code-block:: python

   # PyTorch
   x = torch.randn(4, 16)
   assert torch.allclose(x, x)

   # JAX
   x = random.normal(random.key(0), (4, 16))
   assert jnp.allclose(x, x)

Model Initialization
~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   # PyTorch Geometric
   conv = GCNConv(16, 32)

   # JraphX
   conv = GCNConv(16, 32, rngs=nnx.Rngs(0))

Testing Strategy
~~~~~~~~~~~~~~~~
- Maintain original test logic and assertions
- Comment out unportable sections with clear TODO notes
- Document conversion limitations in test docstrings

Future Work
-----------

This document should be updated as new features are added to JraphX. Priority should be given to:

1. **Core Convolution Layers** - APPNP, Chebyshev, Graph convolutions
2. **Advanced Pooling** - TopK, SAG, hierarchical pooling
3. **Heterogeneous Support** - Multi-relation and multi-node-type graphs
4. **Advanced Aggregation** - Beyond basic scatter operations

Each new feature implementation should be accompanied by converted tests from the corresponding PyG test files.

Contributing Test Conversions
------------------------------

When contributing new test conversions:

1. Follow the established conversion patterns above
2. Document any limitations or missing functionality
3. Update this document with the conversion status
4. Ensure tests pass with ``python -m pytest``
5. Add TODO comments for unportable test sections

For questions about test conversion priorities or implementation approaches, refer to the main JraphX documentation and the ``docs/source/missing_features.rst`` file.
