Changelog
=========

Version 0.1.0 (unreleased)
--------------------------

This release brings a large number of layers, utilities and data structures in line
with :obj:`torch_geometric` semantics. Numerical outputs, module state layouts and a
few public signatures change as a result, so upgrading from 0.0.4 requires the
migrations listed below.

**Breaking Changes -- Dependencies**

* Minimum dependency versions are raised to ``flax>=0.12.8``, ``jax>=0.10.0``,
  ``jaxlib>=0.10.0`` and ``numpy>=2.0``. The previous floors were not installable as
  declared: ``flax`` only publishes a lower bound on ``jax``, so ``flax==0.12.0`` would
  resolve against a current ``jax`` and then fail at import. ``flax>=0.12.1`` is the
  first release providing ``nnx.Variable.get_value()``/``set_value()``, which this
  library needs for variables that may hold :obj:`None`; the floor is set at the version
  CI actually exercises. ``jax`` 0.10 in turn requires ``numpy>=2.0``, so the old
  ``numpy>=1.21`` was never satisfiable alongside it.
* A ``test-minimum-versions`` CI job installs these exact lower bounds and runs the full
  suite against them, so the declared floor cannot drift from the tested one.

**Breaking Changes -- Public API**

* ``rngs`` is a required keyword-only argument of every layer that owns parameters:
  ``GCNConv``, ``GATConv``, ``GATv2Conv``, ``SAGEConv``, ``TransformerConv``, ``MLP``,
  ``BasicGNN`` and its ``GCN``/``GAT``/``GraphSAGE``/``GIN`` specialisations, and
  ``BasicGNN.init_conv``. Every one of these already failed without it, with an
  unhelpful ``AttributeError: 'NoneType' object has no attribute 'params'`` raised from
  inside layer construction; the failure is now a ``TypeError`` at the call site naming
  the missing argument. ``TransformerConv`` additionally used to fall back to a fixed
  ``nnx.Rngs(0)``, so two layers built without ``rngs`` silently shared an
  initialization -- that fallback is gone.
* ``JumpingKnowledge(mode="lstm")`` raises ``ValueError`` when ``num_features``,
  ``num_layers`` or ``rngs`` is missing, instead of ``AssertionError`` (which vanishes
  under ``python -O``) or, for ``rngs``, an ``AttributeError``. Layers whose parameters
  do not need a key -- ``BatchNorm``, ``LayerNorm``, ``GraphNorm``, ``GINConv``,
  ``EdgeConv``, and ``JumpingKnowledge`` in ``cat``/``max`` mode -- still accept none.
* ``TransformerConv.message`` is renamed to ``_attention_message``. It never
  implemented the :meth:`MessagePassing.message` contract: it takes projected
  query/key/value tensors and is not reached through ``propagate``.
* ``GCNConv(cached=True)`` no longer fills its cache on the first forward pass. Call
  ``conv.precompute_norm(edge_index, edge_weight=None, num_nodes=None)`` once, outside
  of any JAX transformation, before the first forward pass; otherwise the layer raises
  a ``RuntimeError``. Use ``conv.reset_cache()`` before re-running ``precompute_norm``
  for a different graph, or keep ``cached=False`` to normalize on every call.
* ``MessagePassing.propagate`` follows the PyG bipartite convention: ``x`` is either a
  single feature table or an ``(x_src, x_dst)`` tuple, where ``x_src`` holds the source
  set and ``x_dst`` the target set. The output has one row per target node. Passing a
  tuple whose sizes disagree with an explicit ``size`` raises a ``ValueError``.
* ``MessagePassing.message`` is invoked exactly once per forward pass.
  ``MessagePassing.message_and_aggregate`` is now an opt-in fused hook: the base class
  raises ``NotImplementedError``, and ``propagate`` dispatches to it only for subclasses
  that override it. Its first argument is the node feature table (or bipartite tuple),
  not the pre-computed messages.
* ``Batch`` is a valid JAX pytree. Its batching configuration --
  ``NODE_INDEX_FIELDS``, ``ELEMENT_LEVEL_FIELDS``, ``GRAPH_LEVEL_FIELDS`` and
  ``_DATA_CLASS`` -- are ``ClassVar`` class attributes rather than dataclass fields, so
  they no longer appear in ``Batch.__init__``, in ``dataclasses.fields(Batch)`` or in
  the pytree. Replace ``Batch(x=..., NODE_INDEX_FIELDS={'face'})`` with a ``Batch``
  subclass that declares ``NODE_INDEX_FIELDS: ClassVar[set[str]] = {'face'}`` in its
  class body.
* ``to_edge_index`` always returns an edge attribute array; it never returns
  :obj:`None` for the second element.
* ``to_undirected`` coalesces its result: the returned edge list is row-wise sorted,
  duplicated edges appear once, and their features are merged with ``reduce``. The
  number of edges is therefore data-dependent and the function cannot be traced by
  :obj:`jax.jit`.
* ``coalesce`` and ``to_undirected`` accept ``reduce="sum"`` as an alias of
  ``reduce="add"``. ``scatter``, on the other hand, rejects unknown reductions with an
  explicit ``ValueError``; only ``"add"``/``"sum"``, ``"mean"``, ``"max"`` and
  ``"min"`` are supported (``"mul"`` was never implemented).
* ``TopKPooling``/``SAGPooling`` interpret ``ratio`` by type, matching PyG: a
  :obj:`float` keeps :math:`\lceil \mathrm{ratio} \cdot N_i \rceil` nodes per graph and
  an :obj:`int` keeps exactly that many. ``ratio=2.0`` previously kept two nodes and
  now keeps every node; write ``ratio=2`` for the old behaviour.
* Pooling is explicit about traceability. ``global_add_pool``, ``global_mean_pool``,
  ``global_max_pool`` and ``global_min_pool`` raise a ``ValueError`` when ``batch`` is a
  tracer and ``size`` is omitted -- pass ``size=<num_graphs>`` inside :obj:`jax.jit` or
  :obj:`jax.vmap`. ``TopKPooling`` and ``SAGPooling`` select a data-dependent number of
  nodes and cannot be traced at all. ``LayerNorm(mode="graph")`` and ``GraphNorm``
  likewise need an explicit ``batch_size`` under a trace.

**Breaking Changes -- Module State Layout**

Checkpoints written by 0.0.4 are not loadable as-is:

* ``GCNConv`` builds its inner ``Linear`` with ``use_bias=False`` and holds its own
  bias, which is added after aggregation. The state key moves from ``linear.bias`` to
  ``bias``.
* ``GraphNorm`` gains a ``mean_scale`` parameter, initialized to ones.
* ``BatchNorm`` stores ``running_mean``, ``running_var`` and ``num_batches_tracked`` as
  ``nnx.BatchStat`` instead of ``nnx.Variable``, so ``nnx.split`` and
  ``nnx.state(..., nnx.Param)`` partition them differently.

**Breaking Changes -- Numerics**

Trained weights still load (subject to the state-layout notes above), but outputs move:

* ``GATConv``/``GATv2Conv`` normalize attention coefficients per head. Outputs change
  for ``heads > 1``.
* ``TransformerConv`` adds the projected edge features to the keys as well as to the
  values, so edge information conditions the attention scores.
* ``GCNConv`` normalizes by weighted degree (rather than by edge count) and adds its
  bias after aggregation.
* ``GraphNorm`` computes per-feature, per-graph statistics and applies the learnable
  ``mean_scale``.
* ``LayerNorm(mode="graph")`` reduces over both the node axis and the feature axis for
  each graph, making it genuinely distinct from ``mode="node"``.
* ``BatchNorm`` pools statistics over every node of the mini-batch and ignores the
  ``batch`` vector, which it previously used to average per-graph statistics. Its
  running variance now tracks the unbiased estimator, matching PyTorch. Use
  ``GraphNorm`` when per-graph statistics are wanted.
* ``TopKPooling``/``SAGPooling`` always apply the score gate to the pooled features, so
  the scoring projection receives gradient. ``multiplier`` is applied after the gate and
  does not influence node selection.
* ``JumpingKnowledge(mode="lstm")`` computes the correct bidirectional GRU recurrence,
  so its outputs change.
* ``scatter_std`` applies Bessel's correction by default, matching ``torch_scatter``.
  Pass ``unbiased=False`` for the previous population standard deviation.
* ``scatter_max``/``scatter_min`` preserve integer input dtypes instead of promoting to
  float.
* ``GCNConv`` matches :obj:`torch_geometric` on graphs that already contain self-loops:
  an existing loop keeps its own weight and is counted once in the degree instead of
  being duplicated alongside an injected unit loop. To keep the output shape static and
  therefore traceable, the duplicated row remains in the returned ``edge_index`` with
  weight ``0.0``; every coefficient and the convolution output agree with PyG.
* ``TopKPooling(min_score=...)`` thresholds and gates with the softmax of the
  unnormalized projection :math:`Xp`, matching PyG's ``SelectTopK``. The projection was
  previously divided by :math:`\lVert p \rVert` first, which rescaled the logits and
  collapsed the gate to nearly one-hot at initialization. The ``ratio`` path is
  unchanged and still normalizes by :math:`\lVert p \rVert`.

**Bug Fixes**

* ``global_max_pool``/``global_min_pool``/``global_mean_pool`` keep the rank of the node
  features: a 1-D input ``[num_nodes]`` pools to ``[batch_size]``, and inputs such as
  ``[num_nodes, heads, features]`` pool to ``[batch_size, heads, features]``.
* ``scatter_log_softmax`` returns ``-inf`` rather than ``NaN`` for a group whose entries
  are all ``-inf``, so ``exp()`` of the result matches ``scatter_softmax``.
* ``coalesce`` accumulates edge identifiers in ``int64``, fixing silent overflow on
  graphs with more than roughly 46k nodes.
* ``Batch.from_data_list`` rejects an attribute that is present on only some graphs when
  that attribute aligns with the batch vector, and accepts one that aligns with the edge
  or element axis -- an edgeless graph may sit alongside graphs carrying ``edge_attr``.
* ``Batch.to_data_list`` preserves trailing graphs that contain no nodes, and keeps the
  leading dimension of a graph-level ``y``.
* ``SAGEConv`` accepts the ``(x_src, None)`` bipartite pair again, and ``EdgeConv`` and
  ``GINConv`` accept the ``(x_src, x_dst)`` tuples their docstrings advertise.
* ``add_self_loops`` with a string ``fill_value`` no longer requires ``num_nodes``.

**Other Changes**

* ``mypy src/`` passes under the project's strict configuration and is a required CI
  check. It previously had no execution path at all -- no CI job, no ``typecheck`` Make
  rule -- and reported 118 errors, among them the missing-``rngs`` crashes above.

* ``GCN`` gains ``precompute_norm(edge_index, edge_weight=None, num_nodes=None)``, which
  fills the cache of every ``GCNConv`` layer at once. It must be called eagerly before
  the first forward pass of a ``GCN(cached=True)``.
* Passing ``edge_weight`` to a ``BasicGNN`` whose ``supports_edge_weight`` is
  :obj:`False` (or ``edge_attr`` when ``supports_edge_attr`` is :obj:`False`) raises a
  ``ValueError`` instead of silently dropping the argument. Custom ``BasicGNN``
  subclasses that forward edge information must set the corresponding class attribute.

* The ``batch_size`` argument of ``BasicGNN.__call__`` (and therefore ``GCN``, ``GAT``,
  ``GraphSAGE``, ``GIN``) and of ``MLP.__call__`` is forwarded to the ``layer_norm`` and
  ``graph_norm`` layers instead of being ignored. Supply it as a Python :obj:`int` when
  such a model is traced together with a ``batch`` vector.
* The deprecated Flax ``.value`` accessor is gone throughout the library. Use
  ``variable[...]`` for array-valued ``nnx.Variable`` objects and
  ``variable.get_value()``/``variable.set_value(x)`` for variables that may hold
  :obj:`None`.

Version 0.0.4
-------------

**Breaking Changes**

* Updated minimum Flax requirement to 0.12.0 for improved pytree handling:
* * Now uses ``nnx.List`` for module lists
* * Now uses ``nnx.data(None)`` for optional module attributes

Version 0.0.3
-------------

Initial release of JraphX.

Features
~~~~~~~~

**Core Data Structures**

* ``Data`` class: Single graph representation with node features, edge indices, edge attributes, and graph-level properties
* ``Batch`` class: Efficient batching of multiple graphs into disconnected graph batches with automatic indexing management

**Message Passing Framework**

* Unified ``MessagePassing`` base class providing a standardized interface for all graph neural network layers
* Flexible message computation, aggregation (sum, mean, max, min), and node update functions
* Support for both node-to-node and edge-enhanced message passing paradigms

**Graph Convolution Layers**

* ``GCNConv``: Graph Convolutional Network with spectral-based convolution and optional edge weights
* ``GATConv``: Graph Attention Network with multi-head attention mechanism and learnable attention weights
* ``GATv2Conv``: Improved Graph Attention Network with enhanced attention computation for better expressivity
* ``GraphSAGE`` (``SAGEConv``): GraphSAGE with multiple aggregation functions (mean, max, LSTM) for inductive learning
* ``GINConv``: Graph Isomorphism Network with theoretical guarantees for graph representation power
* ``EdgeConv``: Dynamic edge convolution for learning on point clouds and dynamic graph construction
* ``DynamicEdgeConv``: Enhanced EdgeConv with k-nearest neighbor graph construction
* ``TransformerConv``: Graph Transformer layer with optimized query-key-value projections and positional encodings

**Pooling Operations**

* **Global pooling**: ``global_add_pool``, ``global_mean_pool``, ``global_max_pool``, ``global_min_pool`` for graph-level representations
* **Advanced pooling**: ``global_softmax_pool``, ``global_sort_pool`` for differentiable and sorted aggregations
* **Hierarchical pooling**: ``TopKPooling`` and ``SAGPooling`` for coarsening graph structures with learnable node selection
* **Batched operations**: Optimized versions (``batched_global_*_pool``) for efficient parallel processing of graph batches

**Utility Functions**

* **Scatter operations**: Comprehensive set including ``scatter_add``, ``scatter_mean``, ``scatter_max``, ``scatter_min``, ``scatter_std``, ``scatter_logsumexp`` for flexible aggregation
* **Scatter softmax**: ``scatter_softmax``, ``scatter_log_softmax``, ``masked_scatter_softmax`` for attention-like mechanisms
* **Graph utilities**: Degree computation (``degree``, ``in_degree``, ``out_degree``), self-loop management (``add_self_loops``, ``remove_self_loops``)
* **Conversion functions**: ``to_dense_adj``, ``to_edge_index``, ``to_undirected`` for different graph representations
* **Graph preprocessing**: ``coalesce`` for edge deduplication, ``maybe_num_nodes`` for automatic node count inference

**Pre-built Models**

* ``GCN``, ``GAT``, ``GraphSAGE``, ``GIN``: Complete model implementations with configurable depth, hidden dimensions, and activation functions
* ``JumpingKnowledge``: Multi-layer aggregation with concatenation, max, and LSTM-based combination strategies
* ``MLP``: Multi-layer perceptron with dropout, batch normalization, and flexible activation functions
* ``BasicGNN``: Abstract base class for implementing custom GNN architectures with standardized interfaces

**Normalization Layers**

* ``BatchNorm``: Batch normalization with running statistics for stable training across graph batches
* ``LayerNorm``: Layer normalization supporting both node-wise and graph-wise normalization schemes
* ``GraphNorm``: Graph-specific normalization designed for graph neural network architectures

**JAX Integration & Performance**

* Extensive use of ``jax.vmap`` and ``nnx.vmap`` for efficient parallel processing of graph batches
* Memory-efficient training patterns using ``jax.lax.scan`` and ``nnx.scan`` for sequential operations
* JIT compilation support for all operations with optimized JAX primitives
* Efficient scatter operations using JAX's advanced indexing (``at[].add/max/min``) for high-performance aggregation
