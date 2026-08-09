Creating Message Passing Networks with JAX
==========================================

Generalizing the convolution operator to irregular domains is typically expressed as a *neighborhood aggregation* or *message passing* scheme.
With :math:`\mathbf{x}^{(k-1)}_i \in \mathbb{R}^F` denoting node features of node :math:`i` in layer :math:`(k-1)` and :math:`\mathbf{e}_{j,i} \in \mathbb{R}^D` denoting (optional) edge features from node :math:`j` to node :math:`i`, message passing graph neural networks can be described as

.. math::
  \mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \bigoplus_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right),

where :math:`\bigoplus` denotes a differentiable, permutation invariant function, *e.g.*, sum, mean or max, and :math:`\gamma` and :math:`\phi` denote differentiable functions such as MLPs (Multi Layer Perceptrons).

In **JraphX**, we implement this using JAX and Flax NNX for neural network components.

.. contents::
    :local:

The "MessagePassing" Base Class
-------------------------------

**JraphX** provides the :class:`~jraphx.nn.conv.message_passing.MessagePassing` base class, which helps in creating such kinds of message passing graph neural networks by automatically taking care of message propagation.
The user only has to define the functions :math:`\phi` (*i.e.* :meth:`~jraphx.nn.conv.message_passing.MessagePassing.message`), and :math:`\gamma` (*i.e.* :meth:`~jraphx.nn.conv.message_passing.MessagePassing.update`), as well as the aggregation scheme to use (*i.e.* :obj:`aggr="add"`, :obj:`aggr="mean"` or :obj:`aggr="max"`).

This is done with the help of the following methods:

* :obj:`MessagePassing(aggr="add", flow="source_to_target")`: Defines the aggregation scheme to use (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`) and the flow direction of message passing (either :obj:`"source_to_target"` or :obj:`"target_to_source"`).
* :obj:`MessagePassing.propagate(edge_index, x, edge_attr=None, size=None)`:
  The initial call to start propagating messages.
  Takes in the edge indices, the node features to gather from, and the optional edge features.
  Note that :func:`~jraphx.nn.conv.message_passing.MessagePassing.propagate` is not limited to exchanging messages in square adjacency matrices of shape :obj:`[N, N]` only, but can also exchange messages in general sparse assignment matrices, *e.g.*, bipartite graphs, of shape :obj:`[N, M]` by passing :obj:`size=(N, M)` as an additional argument.
  If set to :obj:`None`, the assignment matrix is assumed to be a square matrix.
  For bipartite graphs with two independent sets of nodes and indices, and each set holding its own information, this split can be marked by passing the information as a tuple, *e.g.* :obj:`x=(x_src, x_dst)`.
* :obj:`MessagePassing.message(x_j, x_i=None, edge_attr=None)`: Constructs messages to node :math:`i` in analogy to :math:`\phi` for each edge :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
  It receives the gathered source features, the gathered target features and the edge features.
  Note that we generally refer to :math:`i` as the central nodes that aggregates information, and refer to :math:`j` as the neighboring nodes, since this is the most common notation.
* :obj:`MessagePassing.update(aggr_out, x)`: Updates node embeddings in analogy to :math:`\gamma` for each node :math:`i \in \mathcal{V}`.
  Takes the output of aggregation as its first argument and the original target-side node features as its second.

.. important::

    **JraphX dispatches to** :meth:`message` **and** :meth:`update` **positionally**, and
    this differs from PyTorch Geometric in two ways that matter when you port a layer.

    There is no signature introspection, so :meth:`propagate` takes exactly the four
    parameters listed above -- passing an extra keyword such as
    :obj:`propagate(edge_index, x=x, norm=norm)` raises a :obj:`TypeError`. Anything else
    a message needs must be carried on the layer itself, folded into ``edge_attr``, or
    applied before or after the call.

    Because the call is positional, the *names* of your :meth:`message` parameters carry
    no meaning: the first parameter always receives the **source** features and the
    second the **target** features, whatever you call them. There is no :obj:`_i`/:obj:`_j`
    suffix convention. Writing :obj:`def message(self, x_i, x_j)` does not swap them --
    it silently binds ``x_i`` to the source and ``x_j`` to the target, which is the
    reverse of what the names say and produces wrong numbers for any message that is not
    symmetric in its two endpoints. Always declare it as
    :obj:`def message(self, x_j, x_i=None, edge_attr=None)`.

Let us verify this by re-implementing two popular GNN variants, the `GCN layer from Kipf and Welling <https://arxiv.org/abs/1609.02907>`_ and the `EdgeConv layer from Wang et al. <https://arxiv.org/abs/1801.07829>`_.

Implementing the GCN Layer
--------------------------

The `GCN layer <https://arxiv.org/abs/1609.02907>`_ is mathematically defined as

.. math::

    \mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}} \cdot \left( \mathbf{W}^{\top} \cdot \mathbf{x}_j^{(k-1)} \right) + \mathbf{b},

where neighboring node features are first transformed by a weight matrix :math:`\mathbf{W}`, normalized by their degree, and finally summed up.
Lastly, we apply the bias vector :math:`\mathbf{b}` to the aggregated output.
This formula can be divided into the following steps:

1. Add self-loops to the adjacency matrix.
2. Linearly transform node feature matrix.
3. Compute normalization coefficients.
4. Normalize node features in :math:`\phi`.
5. Sum up neighboring node features (:obj:`"add"` aggregation).
6. Apply a final bias vector.

Steps 1-3 are typically computed before message passing takes place.
Steps 4-5 can be easily processed using the :class:`~jraphx.nn.conv.message_passing.MessagePassing` base class.
The full layer implementation is shown below:

.. code-block:: python

    import jax.numpy as jnp
    from flax import nnx
    from jax.ops import segment_sum
    from jraphx.nn.conv.message_passing import MessagePassing
    from jraphx.utils import add_self_loops, degree

    class GCNConv(MessagePassing):
        def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
            super().__init__(aggr='add')  # "Add" aggregation (Step 5).
            # No bias on the linear map: the bias belongs *after* aggregation (Step 6).
            # Folding it in here would let the degree normalization rescale it, which is
            # not what the equation above says.
            self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)
            self.bias = nnx.Param(jnp.zeros((out_features,)))

        def __call__(self, x, edge_index):
            # x has shape [N, in_features]
            # edge_index has shape [2, E]

            # Step 2: Linearly transform node feature matrix first (more efficient).
            x = self.linear(x)

            # Step 1: Add self-loops to the adjacency matrix.
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

            # Step 3: Compute normalization.
            row, col = edge_index[0], edge_index[1]
            deg = degree(col, x.shape[0], dtype=x.dtype)
            deg_inv_sqrt = jnp.power(deg, -0.5)
            deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0.0, deg_inv_sqrt)
            # Create edge weights from normalization
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Step 4-5: Efficient message passing with normalization.
            messages = jnp.take(x, row, axis=0) * edge_weight.reshape(-1, 1)
            out = segment_sum(messages, col, num_segments=x.shape[0])

            # Step 6: Apply the bias to the aggregated result.
            return out + self.bias[...]

:class:`~jraphx.nn.conv.GCNConv` inherits from :class:`~jraphx.nn.conv.message_passing.MessagePassing` with :obj:`"add"` aggregation.
All the logic of the layer takes place in its :meth:`__call__` method.
Here, we first linearly transform node features using :class:`nnx.Linear` (step 2 - done first for better cache efficiency), then add self-loops to our edge indices using :func:`~jraphx.utils.add_self_loops` (step 1).

The normalization coefficients are derived by the node degrees :math:`\deg(i)` for each node :math:`i` which gets transformed to :math:`1/(\sqrt{\deg(i)} \cdot \sqrt{\deg(j)})` for each edge :math:`(j,i) \in \mathcal{E}`.
The result is saved in the array :obj:`edge_weight` of shape :obj:`[num_edges, ]` (step 3).

For efficient computation, we use JAX's optimized operations:
- :func:`jnp.take` for fast indexing to gather source node features
- Element-wise multiplication to apply edge weights
- :func:`jax.ops.segment_sum` for efficient aggregation by target nodes

This approach is more efficient than the traditional :meth:`propagate` method because it directly leverages JAX's optimized array operations.

That is all that it takes to create a simple message passing layer with JAX!
You can use this layer as a building block for deep architectures.
Initializing and calling it is straightforward:

.. code-block:: python

    conv = GCNConv(16, 32, rngs=nnx.Rngs(42))
    output = conv(x, edge_index)

Implementing the Edge Convolution
---------------------------------

The `edge convolutional layer <https://arxiv.org/abs/1801.07829>`_ processes graphs or point clouds and is mathematically defined as

.. math::

    \mathbf{x}_i^{(k)} = \max_{j \in \mathcal{N}(i)} h_{\mathbf{\Theta}} \left( \mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)} - \mathbf{x}_i^{(k-1)} \right),

where :math:`h_{\mathbf{\Theta}}` denotes an MLP.
In analogy to the GCN layer, we can use the :class:`~jraphx.nn.conv.message_passing.MessagePassing` class to implement this layer, this time using the :obj:`"max"` aggregation:

.. code-block:: python

    import jax.numpy as jnp
    from flax import nnx
    from jraphx.nn.conv.message_passing import MessagePassing

    class EdgeConv(MessagePassing):
        def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
            super().__init__(aggr='max')  # "Max" aggregation.
            self.mlp = nnx.Sequential(
                nnx.Linear(2 * in_features, out_features, rngs=rngs),
                nnx.relu,
                nnx.Linear(out_features, out_features, rngs=rngs)
            )

        def __call__(self, x, edge_index):
            # x has shape [N, in_features]
            # edge_index has shape [2, E]

            return self.propagate(edge_index, x)

        def message(self, x_j, x_i, edge_attr=None):
            # x_i has shape [E, in_features]
            # x_j has shape [E, in_features]

            tmp = jnp.concatenate([x_i, x_j - x_i], axis=1)  # tmp has shape [E, 2 * in_features]
            return self.mlp(tmp)

Inside the :meth:`~jraphx.nn.conv.message_passing.MessagePassing.message` function, we use :obj:`self.mlp` to transform both the target node features :obj:`x_i` and the relative source node features :obj:`x_j - x_i` for each edge :math:`(j,i) \in \mathcal{E}`.

The edge convolution is actually a dynamic convolution, which recomputes the graph for each layer using nearest neighbors in the feature space.
**JraphX** provides a :class:`~jraphx.nn.conv.DynamicEdgeConv` implementation that handles this automatically:

.. code-block:: python

    from jraphx.nn.conv import DynamicEdgeConv
    from jraphx.nn.models import MLP

    # Create neural network for edge feature processing
    nn = MLP(feature_list=[6, 128], rngs=nnx.Rngs(42))  # Input: 2*3=6, Output: 128

    # Create dynamic edge convolution layer
    conv = DynamicEdgeConv(
        nn=nn,
        k=6,  # Number of nearest neighbors
    )

    # Use with point cloud data (x contains spatial coordinates)
    # Note: k-NN indices must be pre-computed from spatial coordinates
    output = conv(x, knn_indices=knn_indices)

Note that unlike PyTorch Geometric's version, JraphX's DynamicEdgeConv does not automatically compute k-NN graphs from node features. You must provide the k-NN indices separately, typically computed using external libraries or custom JAX implementations for spatial/feature-space nearest neighbors.

Exercises
---------

Imagine we are given the following :class:`~jraphx.data.Data` object:

.. code-block:: python

    import jax.numpy as jnp
    from jraphx.data import Data

    edge_index = jnp.array([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2, 1]], dtype=jnp.int32)
    x = jnp.array([[-1.0], [0.0], [1.0]], dtype=jnp.float32)

    data = Data(x=x, edge_index=edge_index.T)

Try to answer the following questions related to :class:`~jraphx.nn.conv.GCNConv`:

1. What information does :obj:`row` and :obj:`col` hold in the context of JAX arrays?

2. What does :func:`~jraphx.utils.degree` do and how is it different from PyTorch's version?

3. Why do we use :obj:`degree(col, ...)` rather than :obj:`degree(row, ...)`?

4. What does :obj:`deg_inv_sqrt[col]` and :obj:`deg_inv_sqrt[row]` do in terms of JAX indexing?

5. How does :func:`jnp.take` work in the JraphX implementation compared to PyTorch's automatic lifting?

6. Add an :meth:`~jraphx.nn.conv.MessagePassing.update` function to the custom :class:`GCNConv` that adds transformed central node features to the aggregated output.

7. What are the benefits of using :func:`jax.ops.segment_sum` over the traditional message passing approach?

Try to answer the following questions related to :class:`~jraphx.nn.conv.EdgeConv`:

1. What is :obj:`x_i` and :obj:`x_j - x_i` in the context of JAX arrays?

2. What does :obj:`jnp.concatenate([x_i, x_j - x_i], axis=1)` do? Why :obj:`axis = 1`?

3. Implement a vectorized version of EdgeConv that processes multiple graphs using :obj:`nnx.vmap`.
