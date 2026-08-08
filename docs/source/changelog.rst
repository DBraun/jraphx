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

* ``BatchNorm`` and ``LayerNorm`` no longer accept the arguments they never read:
  ``axis``, ``axis_name``, ``axis_index_groups`` and ``use_fast_variance`` on
  ``BatchNorm``; ``reduction_axes``, ``feature_axes``, ``axis_name``,
  ``axis_index_groups`` and ``use_fast_variance`` on ``LayerNorm``. All of these were
  stored and silently ignored -- in particular ``axis_name``, which read as if it
  synchronized statistics across devices. Passing one now raises ``TypeError``.
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
  now keeps every node; write ``ratio=2`` for the old behavior.
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
* Layers that support dropout always hold a ``Dropout`` submodule; a rate of
  :obj:`0.0` makes it return its input untouched without drawing a key, rather than
  the layer holding :obj:`None`. Every such module therefore carries dropout
  ``RngState`` even at rate 0, so ``jax.grad`` over an unfiltered ``nnx.split(model)``
  now fails on the integer RNG counters. Split the parameters out first::

      graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  Parameter initialization is unchanged: the ``Dropout`` is constructed after the
  layers that draw parameter keys.

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
* ``SAGEConv`` applies its neighbor transform *after* aggregation, as
  :math:`\mathbf{W}_2 \cdot \mathrm{aggr}_j \mathbf{x}_j`. Only ``aggr="max"`` changes:
  an elementwise maximum does not commute with a linear map, so the previous ordering
  computed :math:`\max_j (\mathbf{W}_2 \mathbf{x}_j)` -- a maximum taken in the output
  space, mixing columns drawn from different source nodes. ``aggr="mean"``/``"gcn"`` are
  unaffected, since sum and mean do commute.
* ``DynamicEdgeConv`` builds the k-NN ``edge_index`` with the neighbor as source and the
  querying node as target, so a node aggregates over the neighbors *it* selected. The
  rows were previously the other way round, which built the reverse k-NN graph: because
  "j is among i's k nearest" is not symmetric, every node aggregated over the nodes that
  had selected it, and a node selected by nobody received no messages at all and
  max-aggregated to a zero row. Every ported DGCNN model changes.
* ``GATConv``/``GATv2Conv`` no longer count a pre-existing self-loop twice. PyG removes
  self-loops before inserting its own; without that removal a node arriving with a loop
  got two, roughly doubling its self-attention mass and correspondingly down-weighting
  its real neighbors. Dropping the duplicate column would make the edge count
  data-dependent and break :obj:`jax.jit`, so its attention logit is driven to
  :math:`-\infty` instead, which is exactly a softmax weight of zero. The duplicate row
  stays in the ``edge_index`` returned by ``return_attention_weights=True``, carrying a
  weight of zero. With a string ``fill_value`` the generated loop features still reduce
  over a set that includes the original loop, which PyG excludes.
* ``GATConv``/``GATv2Conv`` size their self-loop set from
  ``min(num_src_nodes, num_dst_nodes)`` on a bipartite graph, matching PyG: a self-loop
  only exists for a node present in both endpoint tables. Sizing it from the target count
  alone appended loops whose source index was out of range, and JAX's array indexing
  clamps such an out-of-bounds gather to the last row rather than raising -- several
  target nodes received the same fabricated message, and a target with no incoming edge
  acquired a value. Both
  layers now also validate their gather indices and raise :obj:`IndexError`, which
  ``propagate`` already did for the layers that route through it.
* ``scatter_mean``, ``scatter_std``, ``segment_mean``, ``GraphNorm`` and
  ``LayerNorm(mode="graph")`` accumulate both their running total and their member count
  in at least float32, never in a narrower input dtype. bfloat16 carries 8 mantissa bits,
  so its consecutive integers stop at 256 -- ``256 + 1`` rounds back to 256 -- and both
  accumulators froze partway through any segment larger than that, leaving the quotient
  wrong by a degree-dependent factor. A floating-point caller's dtype is still what comes
  back; integer inputs divide to float32, as :func:`jax.numpy.true_divide` would.
* ``scatter_softmax``, ``scatter_log_softmax`` and ``scatter_logsumexp`` accumulate
  their per-group exponential sums in at least float32 as well. In bfloat16 the running
  normalizer froze at 256, so the attention weights of any larger group -- these
  functions normalize ``GATConv`` and ``TransformerConv`` attention -- summed to
  substantially more than 1.
* ``batch_histogram`` assigns each value to the bin whose interval contains it, matching
  :func:`numpy.histogram`: with an explicit ``min_val``/``max_val`` the values outside
  the range are dropped, not folded into the edge bins. It previously searched only the
  *left* edges and from the left, which pushed every value strictly inside a bin one
  place to the right, left bin 0 collecting only values exactly equal to the lower
  bound, and made the last bin absorb the overflow. One residual divergence: bin edges
  are computed in the working precision (float32), so a value exactly on an interior
  edge can land one bin away from numpy's float64 edges.
* ``TransformerConv`` matches PyG's parameterization: the output projection that PyG
  does not have is removed (the forward now ends at the skip/beta combination, as the
  docstring formula always claimed), the fused query/key/value projection carries a
  bias like PyG's three ``Linear`` layers, heads are concatenated or averaged *before*
  the skip term, and ``lin_skip``/``lin_beta`` are sized for the final output width
  under ``concat=False``. ``beta=True`` now requires ``root_weight=True`` and is
  ignored otherwise -- gating against the node's raw value projection mixed a
  transformed root feature into every row of a layer documented not to have one, and
  gave an isolated node a nonzero output where PyG yields zero.
* The ``GAT`` model concatenates ``hidden_features // heads`` narrow heads on its last
  layer when ``out_features`` is :obj:`None`, as PyG's ``BasicGNN`` does; it previously
  averaged full-width heads into the same output shape, hiding a different architecture
  and parameterization. It also forwards its ``dropout_rate`` into every
  ``GATConv``/``GATv2Conv`` as attention dropout, the GAT paper's primary regularizer.
* The ``GCN``/``GAT``/``GraphSAGE``/``GIN`` models forward their remaining keyword
  arguments to the convolution constructors, so ``GCN(..., bias=False)``,
  ``GIN(..., eps=0.7)`` and ``GAT(..., add_self_loops=False)`` take effect and an
  unsupported argument raises ``TypeError``. All of these were silently discarded.
* ``GATConv``/``GATv2Conv`` with a bipartite ``(x_src, None)`` input omit the target
  attention term, as PyG does. Both layers previously gathered the *source* table at
  target indices to fabricate target features, changing every attention weight.
* ``GATv2Conv`` defaults ``fill_value`` to ``"mean"``, matching PyG and ``GATConv``;
  self-loop edge features were previously zero-filled.
* ``global_sort_pool`` sorts by the **last feature channel** (the DGCNN SortPooling
  operator, PyG's ``SortAggregation``) instead of by the feature sum, and zero-pads
  small graphs *after* selection, so padding can no longer outrank real nodes whose
  sort scores are negative.
* ``in_degree``/``out_degree`` size their result by every node the full ``edge_index``
  mentions. Inferring the count from the counted row alone silently truncated the
  vector whenever a node appeared only at the other endpoint.
* ``Data.is_directed`` compares the lexicographically sorted edge list with its
  reverse, making it exact at any graph size and multiset-correct for duplicated
  edges. The previous ``src * num_nodes + dst`` packing wrapped int32 above ~46k nodes
  -- the same overflow fixed in ``coalesce`` -- and its set semantics called a graph
  with an unbalanced duplicated edge undirected.
* ``add_remaining_self_loops`` gives every node exactly one self-loop: existing loops
  are removed first (collapsing duplicates), the per-node loops are appended in node
  order, and a replaced loop keeps its attribute. Duplicated input loops were
  previously all retained, where PyG collapses them.
* ``TransformerConv(beta=True)`` concatenates its gate input as
  :math:`[\mathbf{m}_i, \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i - \mathbf{W}_1
  \mathbf{x}_i]`, the order PyG's *implementation* uses -- PyG's docstring lists the
  reversed order, which its code does not. Found by the weight-transplant parity
  harness: with the previous order a transplanted ``lin_beta`` computed a different
  gate.

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
* ``Batch`` subclasses with several ``NODE_INDEX_FIELDS`` pick their primary index
  field alphabetically instead of by set iteration order, which varied with the
  per-process hash seed and made the same legitimate data list collate in one process
  and raise ``RuntimeError`` in another.

**Other Changes**

* A parity suite under ``tests/parity/`` transplants weights layer by layer into an
  installed :obj:`torch_geometric` and compares outputs elementwise -- convolutions,
  normalization layers, scatter/loop/graph utilities and poolings. A dedicated CI job
  installs CPU ``torch`` and ``torch_geometric`` and gates releases on it; the jobs
  without torch skip the package. The one deliberate divergence (string
  ``fill_value`` self-loop features on a graph that already carries a loop) is pinned
  as a strict ``xfail``.
* The ``docs`` extra installs on Python 3.13: ``sphinx==5.1.1`` and
  ``sphinx-autodoc-typehints==1.19.2`` -- neither importable there -- are replaced by
  floors on current releases, and the documentation builds warning-free against
  sphinx 9.
* The sdist no longer ships ``tests/test_version.py``, which setuptools' legacy
  default template included on its own. It was the only test file in the archive and
  could not pass from an sdist install, since it reads the changelog out of the
  excluded ``docs/`` tree.
* New :class:`~jraphx.nn.conv.GINEConv` layer -- :class:`~jraphx.nn.conv.GINConv` with
  edge features fused into every message, from `"Strategies for Pre-training Graph
  Neural Networks" <https://arxiv.org/abs/1905.12265>`_ -- contributed by
  `@jiinyih <https://github.com/jiinyih>`_. Edge features of a different width than the
  nodes are projected via ``edge_dim``; without ``edge_dim`` a width mismatch raises
  ``ValueError``, as in PyG, rather than silently broadcasting. Supports bipartite
  ``(x_src, x_dst)`` input.
* New :func:`~jraphx.utils.parse_dtype` utility resolving a dtype spec -- a plain or
  prefixed string (``"float32"``, ``"jnp.bfloat16"``, ``"np.int32"``), a scalar type,
  or a dtype object -- to the matching jax.numpy scalar type. Every jraphx ``dtype``
  argument (``degree``/``in_degree``/``out_degree``, ``GCNConv`` normalization,
  ``BatchNorm``/``LayerNorm`` ``dtype``/``param_dtype``) now routes through it, so a
  dtype can come straight from a configuration file and an invalid spec fails at the
  call site with a clear error. Abstract categories such as ``"floating"`` are
  rejected there rather than surfacing later at the first array construction.
* Type annotations use :obj:`jax.Array`, the canonical public name of the array type,
  instead of the ``jnp.ndarray`` alias -- in the library, the tests and the
  documentation. No runtime behavior changes.
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
* A ``NOTICE`` file collects the third-party notices the project's own README already
  claimed, including the MIT permission notice for the PyTorch Geometric code and
  docstrings this library derives from. It is listed in ``license-files``, so it ships in
  both the wheel and the sdist.
* Documentation corrections. Three pages taught a training step wrapped in
  :func:`jax.jit`, where the parameter update is traced on a copy of the module state and
  silently discarded -- no error, a plausible loss, and not one parameter moved. They now
  use :func:`nnx.jit`, and :doc:`/advanced/jit` explains when each is appropriate --
  including the :func:`nnx.split`/:func:`nnx.merge` functional training loop, under
  which :func:`jax.jit` is correct because the state is threaded explicitly, and which
  the Flax performance guide recommends for hot loops. The
  ``MessagePassing`` prose described PyG's ``propagate(**kwargs)`` and ``_i``/``_j``
  argument lifting, neither of which JraphX implements: ``message`` is dispatched
  positionally, so a signature written ``message(self, x_i, x_j)`` binds the *source*
  features to ``x_i``. Two shipped examples did exactly that. Also fixed: the two
  ``Batch``-subclass recipes, which listed node-level fields as ``ELEMENT_LEVEL_FIELDS``
  and raised; a ``BatchNorm(affine=True)`` snippet, since the argument is spelled
  ``use_scale``/``use_bias``; a reference to a ``jraphx.data.vmap_batch`` module that does
  not exist; and a call to the removed ``jax.tree_map``.
* Example corrections. ``examples/gcn_jraphx.py`` and ``examples/gcn_standalone.py``
  sharded one graph's nodes against a slice of its edges under ``shard_map``, so edges
  carrying global node ids gathered out of range and every message crossing a device
  boundary was dropped -- silently, producing plausible but wrong losses and accuracies.
  Both now shard whole graphs, and the sharded result matches the unsharded one exactly.
  ``examples/nnx_transforms.py`` passed its model and optimizer to ``nnx.scan`` as
  broadcast inputs, whose mutations Flax discards, so its "memory-efficient training"
  never updated a parameter; it now steps through the mini-batches and asserts that the
  loss falls.
* A third documentation pass, executing every snippet it touched. The flagship
  :doc:`/get_started/introduction` training and evaluation examples boolean-indexed
  traced arrays under :func:`nnx.jit` and crashed on their first call; they now weight
  the per-node loss by the mask. :doc:`/modules/pooling` claimed ``TopKPooling`` is
  JIT-compatible directly above the note explaining why it is not; the section now
  pools eagerly and jits the dense computation after it. Also fixed: a
  ``jraphx.data.DataLoader`` import that does not exist, a ``DeepGNN`` example calling
  its ``Data``-taking model with two arrays, a vmap snippet feeding 16-feature graphs
  to the page's 2-feature model, four constructor calls missing the required ``rngs``,
  a snippet using ``jax.random`` without importing it, and docstrings advertising an
  ``aggr="lstm"`` that raises ``NotImplementedError``, a bipartite ``TransformerConv``
  input the code cannot accept, a ``TopKPooling`` formula applying its score
  nonlinearity twice, and a ``JumpingKnowledge`` "bi-directional LSTM" that is
  implemented as two GRU cells.

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
