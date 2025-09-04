:github_url: https://github.com/DBraun/jraphx

JraphX Documentation
====================

**JraphX** is a Graph Neural Network (GNN) library for JAX/Flax NNX, designed as an unofficial successor to DeepMind's archived `jraph <https://github.com/google-deepmind/jraph>`_ library. It provides a `PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric/>`_-inspired API while leveraging the JAX ecosystem's strengths in JIT compilation, `sharding <https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html>`_, and more.

.. note::
   **Attribution Notice**: JraphX builds upon and incorporates code from multiple open-source projects:

   - **PyTorch Geometric** (MIT License, Copyright (c) 2023 PyG Team): JraphX contains substantial portions of code and documentation derived from PyTorch Geometric.
   - **Flax** (Apache License 2.0): The Flax NNX library made JraphX's implementation significantly easier.
   - **Jraph** (Apache License 2.0): DeepMind's original JAX GNN library, which is now archived.

   We are grateful to all development teams for creating these foundational libraries that make JraphX possible.

JraphX consists of various methods for deep learning on graphs and other irregular structures, implementing core GNN layers and utilities with JAX and Flax/NNX. It features efficient message passing frameworks, vectorized operations using `nnx.vmap`, sequential processing with `nnx.scan`, and seamless integration with the JAX ecosystem including automatic differentiation and JIT compilation.

.. toctree::
   :maxdepth: 1
   :caption: Install JraphX

   install/installation

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/introduction
   modules/data

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/create_gnn
   tutorial/gnn_design
   tutorial/examples

.. toctree::
   :maxdepth: 1
   :caption: Advanced Concepts

   advanced/batching
   advanced/jit
   advanced/compile
   advanced/techniques

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   modules/root
   modules/nn
   modules/utils

.. toctree::
   :maxdepth: 1
   :caption: Cheatsheets

   cheatsheet/gnn_cheatsheet
   cheatsheet/data_cheatsheet

.. toctree::
   :maxdepth: 1
   :caption: External Resources

   external/resources
   missing_features
   missing_tests

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   citation
   changelog
