Missing Features in JraphX
==========================

This document tracks PyTorch Geometric features that are not yet implemented in JraphX.

High Priority (Core GNN functionality)
--------------------------------------

Convolution Layers
~~~~~~~~~~~~~~~~~~

- **AGNNConv** - Attention-based Graph Neural Network
- **APPNP** - Approximate Personalized Propagation of Neural Predictions
- **ARMAConv** - ARMA filters on graphs
- **CGConv** - Crystal Graph Convolutional Networks
- **ChebConv** - Chebyshev spectral graph convolution
- **ClusterGCNConv** - Cluster-GCN
- **DNAConv** - Dynamic Network Architecture
- **FastRGCNConv** - Fast Relational Graph Convolutional Networks
- **FeaStConv** - Feature-Steered Convolution
- **FiLMConv** - Feature-wise Linear Modulation
- **GCN2Conv** - Simple and Deep Graph Convolutional Networks
- **GENConv** - Generalized Graph Convolutional Networks
- **GeneralConv** - General GNN layer
- **GPSConv** - Graph Transformer with Positional and Structural Encoding
- **GravNetConv** - GravNet layer for point clouds
- **HGTConv** - Heterogeneous Graph Transformer
- **HypergraphConv** - Hypergraph Convolution
- **LEConv** - Local Extremum Graph Neural Networks
- **LGConv** - Light Graph Convolution
- **MFConv** - Molecular Fingerprint Convolution
- **NNConv** - Continuous kernel-based convolution
- **PANConv** - Path Augmented Graph Neural Networks
- **PDNConv** - Pathfinder Discovery Networks
- **PNAConv** - Principal Neighbourhood Aggregation
- **PointConv** - Point Convolution for 3D
- **PPFConv** - Point Pair Feature Convolution
- **RGCNConv** - Relational Graph Convolutional Networks
- **RGATConv** - Relational Graph Attention Networks
- **ResGatedGraphConv** - Residual Gated Graph ConvNets
- **SGConv** - Simplifying Graph Convolutional Networks
- **SignedConv** - Signed Graph Convolutional Networks
- **SplineConv** - Spline-based convolution
- **SuperGATConv** - SuperGAT
- **TAGConv** - Topology Adaptive Graph Convolutional Networks
- **TWirls** - Trainable Wishart Relational Networks
- **XConv** - PointNet++ XConv layer

Aggregation Functions
~~~~~~~~~~~~~~~~~~~~~

- **Aggregation Module** - Advanced aggregation functions
- **MultiAggregation** - Multiple aggregation combination
- **AttentionalAggregation** - Attention-based aggregation
- **DeepSetsAggregation** - DeepSets aggregation
- **DegreeScalerAggregation** - Degree-based scaling
- **EquilibriumAggregation** - Equilibrium-based aggregation
- **GraphMultisetTransformer** - Graph Multiset Transformer
- **LSTMAggregation** - LSTM-based aggregation
- **MLPAggregation** - MLP aggregation
- **PowerMeanAggregation** - Power mean aggregation
- **Set2Set** - Set2Set aggregation
- **SoftmaxAggregation** - Softmax aggregation
- **SortAggregation** - Sort aggregation
- **VarAggregation** - Variance aggregation

Medium Priority (Advanced Features)
-----------------------------------

Pooling Layers
~~~~~~~~~~~~~~

- **ASAPooling** - Adaptive Structure Aware Pooling
- **EdgePooling** - Edge-based pooling
- **GCNPool** - GCN-based pooling
- **GlobalAttention** - Global attention pooling
- **GraphSAINTSampler** - GraphSAINT sampling
- **HitAndRun** - Hit and Run sampling
- **MaxPooling** - Max pooling on graphs
- **MemPooling** - Memory-based pooling
- **NodeSAINTSampler** - Node-based GraphSAINT sampling
- **PANPooling** - Path Augmented Pooling

Pre-built Models
~~~~~~~~~~~~~~~~

- **AttentiveFP** - Attentive Fingerprinting
- **BASIC_GNN** - Enhanced basic model variations
- **DeepGCN** - Deep Graph Convolutional Networks
- **DeepGraphInfomax** - Deep Graph Infomax
- **DiffPool** - Differentiable Pooling
- **GAE** - Graph Autoencoders
- **VGAE** - Variational Graph Autoencoders
- **GCN** - Enhanced versions
- **GraphSAGE** - Enhanced versions
- **GraphUNet** - Graph U-Net
- **JK-Net** - Enhanced Jumping Knowledge Networks
- **MetaPath2Vec** - MetaPath2Vec for heterogeneous graphs
- **Node2Vec** - Node2Vec embeddings
- **PNA** - Principal Neighbourhood Aggregation networks
- **SchNet** - SchNet for molecular property prediction
- **TGN** - Temporal Graph Networks

Normalization Layers
~~~~~~~~~~~~~~~~~~~~

- **DiffGroupNorm** - Differentiable Group Normalization
- **InstanceNorm** - Instance Normalization
- **MessageNorm** - Message Normalization
- **PairNorm** - Pair Normalization

JAX/JraphX Specific Limitations
-------------------------------

k-NN Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~

- **torch-cluster integration** - PyTorch Geometric's DynamicEdgeConv uses `torch_cluster.knn()` for automatic k-nearest neighbor computation from node features. JraphX's DynamicEdgeConv is a simplified version that requires pre-computed k-NN indices as input.
- **Dynamic graph construction** - Full dynamic graph construction would require a JAX-native k-NN implementation, which is not currently available.

Lower Priority (Specialized Features)
-------------------------------------

Knowledge Graph Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ComplEx** - Complex embeddings
- **DistMult** - DistMult embeddings
- **HolE** - Holographic embeddings
- **KGEModel** - Base class for KG embeddings
- **PairRE** - Paired relation embeddings
- **RotatE** - Rotation-based embeddings
- **TransE** - Translation embeddings

Dense Layers
~~~~~~~~~~~~

- **DenseGCNConv** - Dense GCN convolution
- **DenseGINConv** - Dense GIN convolution
- **DenseGraphConv** - Dense graph convolution
- **DenseSAGEConv** - Dense SAGE convolution
- **LinearTransformation** - Dense linear layers

Functional Operations
~~~~~~~~~~~~~~~~~~~~~

- **dropout** - Graph-aware dropout
- **gumbel_softmax** - Gumbel softmax for graphs
- **local_graph_clustering** - Local clustering
- **pagerank** - PageRank algorithm
- **subgraph** - Subgraph sampling

Transforms (Not Core but Useful)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **AddSelfLoops** - Add self-loops transform
- **Compose** - Transform composition
- **NormalizeFeatures** - Feature normalization
- **RandomNodeSplit** - Random node splitting
- **RemoveIsolatedNodes** - Remove isolated nodes
- **ToDevice** - Device placement transform
- **ToSparseTensor** - Sparse tensor conversion

Data Loading & Sampling
~~~~~~~~~~~~~~~~~~~~~~~

- **DataLoader** - Graph data loading
- **NeighborSampler** - Neighborhood sampling
- **RandomWalkSampler** - Random walk sampling
- **ShaDowKHopSampler** - ShaDow k-hop sampling

Datasets (Not Applicable - JAX doesn't need this)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

❌ All dataset classes (TUDataset, Planetoid, etc.) - Not relevant for JAX-only library

Distributed Training (Future Consideration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **DistributedSampler** - For future JAX distributed training
- **GraphSAINT** - Distributed sampling strategies

Features Deliberately Omitted
-----------------------------

PyTorch-Specific
~~~~~~~~~~~~~~~~

❌ **DataParallel** - JAX uses different parallelization

❌ **torch.compile** integration - JAX uses jit instead

❌ **SparseTensor** support - JAX has different sparse support

❌ **CUDA-specific** operations - might be heavy lift

Framework-Specific
~~~~~~~~~~~~~~~~~~

❌ **Heterogeneous graphs** - Complex feature, may not fit JAX patterns

❌ **Explainability** modules - Separate concern

❌ **NLP modules** - Out of scope

❌ **Remote backend** - PyG-specific

Implementation Status Legend
----------------------------

- ✅ **Implemented** - Available in JraphX
- **Planned** - Should be implemented
- ❌ **Omitted** - Deliberately not implementing

Removed Documentation Files
---------------------------

The following PyTorch Geometric documentation files have been removed from JraphX as they are not applicable to a JAX-based GNN library:

Advanced Concepts (Removed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **cpu_affinity.rst** - PyTorch-specific CPU affinity settings
- **graphgym.rst** - GraphGym framework (PyTorch ecosystem)
- **hgam.rst** - Heterogeneous Graph Attention Memory (not implemented)
- **remote.rst** - Remote backend for PyTorch Geometric
- **sparse_tensor.rst** - PyTorch sparse tensor integration

Module Documentation (Removed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **contrib.rst** - Community contributions (PyTorch-specific)
- **datasets.rst** - Dataset loading (JraphX uses external datasets)
- **distributed.rst** - Distributed training (PyTorch-specific)
- **explain.rst** - Model explainability (separate concern)
- **graphgym.rst** - GraphGym configuration system
- **loader.rst** - Data loading utilities (not needed for JAX)
- **metrics.rst** - Evaluation metrics (use external libraries)
- **profile.rst** - Performance profiling (JAX has its own tools)
- **sampler.rst** - Graph sampling utilities (not implemented)
- **transforms.rst** - Data transforms (JAX uses functional preprocessing)

Tutorial Documentation (Removed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **application.rst** - Application-specific tutorials
- **compile.rst** - torch.compile integration (JAX uses jit)
- **create_dataset.rst** - Dataset creation (not JraphX's scope)
- **dataset_splitting.rst** - Dataset splitting utilities
- **dataset.rst** - Dataset handling
- **distributed_pyg.rst** - Distributed PyTorch Geometric
- **distributed.rst** - Distributed training
- **explain.rst** - Model explainability
- **graph_transformer.rst** - Advanced transformer architectures (not implemented)
- **heterogeneous.rst** - Heterogeneous graph processing (not implemented)
- **load_csv.rst** - CSV loading utilities
- **multi_gpu_vanilla.rst** - Multi-GPU training (PyTorch-specific)
- **multi_node_multi_gpu_vanilla.rst** - Multi-node training (PyTorch-specific)
- **neighbor_loader.rst** - Neighborhood sampling (not implemented)
- **point_cloud.rst** - Point cloud processing (limited support)
- **shallow_node_embeddings.rst** - Node embedding methods (not implemented)

Rationale for Removal
~~~~~~~~~~~~~~~~~~~~~

- **Framework Mismatch**: PyTorch-specific features that don't apply to JAX
- **Scope Limitation**: JraphX focuses on core GNN layers, not entire ML pipelines
- **Unimplemented Features**: Advanced features not yet available in JraphX
- **External Dependencies**: Features that rely on PyTorch ecosystem

Removed Figure Files
~~~~~~~~~~~~~~~~~~~~

The following figure files from ``docs/source/_figures/`` have been removed as they were not referenced in the JraphX documentation:

- **architecture.pdf** / **architecture.svg** - PyTorch Geometric architecture diagrams
- **dist_part.png** / **dist_proc.png** / **dist_sampling.png** - Distributed training figures (PyTorch-specific)
- **graphgps_layer.png** - GraphGPS layer architecture (not implemented)
- **graphgym_design_space.png** / **graphgym_evaluation.png** / **graphgym_results.png** - GraphGym framework figures
- **hg_example.svg** / **hg_example.tex** - Heterogeneous graph examples (not implemented)
- **intel_kumo.png** - Intel optimization figures (not applicable)
- **meshcnn_edge_adjacency.svg** - MeshCNN figures (not implemented)
- **point_cloud1.png** - **point_cloud4.png** - Point cloud examples (limited support)
- **remote_1.png** - **remote_3.png** - Remote backend figures (not applicable)
- **shallow_node_embeddings.png** - Node embedding figures (not implemented)
- **to_hetero.svg** / **to_hetero.tex** / **to_hetero_with_bases.svg** / **to_hetero_with_bases.tex** - Heterogeneous graph conversion (not implemented)
- **training_affinity.png** - CPU affinity training (PyTorch-specific)

Kept Figure Files
~~~~~~~~~~~~~~~~~

Only the essential figures were retained:

- **graph.svg** / **graph.tex** - Basic graph visualization used in introduction tutorial
- **build.sh** - Figure generation script

Kept Documentation Files
~~~~~~~~~~~~~~~~~~~~~~~~

The following files were retained and translated to JraphX:

- **Core tutorials**: ``create_gnn.rst``, ``gnn_design.rst`` (JAX integration)
- **Essential concepts**: ``batching.rst``, ``jit.rst``, ``compile.rst``
- **API reference**: ``nn.rst``, ``data.rst``, ``utils.rst``, ``root.rst``
- **Cheatsheets**: ``gnn_cheatsheet.rst``, ``data_cheatsheet.rst``
- **Getting started**: ``introduction.rst``, ``installation.rst``

Notes
-----

- Priority is based on common usage patterns and core GNN functionality
- JAX-specific optimizations should be added where applicable (jit, vmap, scan)
- Some features may need significant adaptation for JAX/NNX paradigms
- Documentation cleanup focused on maintaining only relevant, translated content
