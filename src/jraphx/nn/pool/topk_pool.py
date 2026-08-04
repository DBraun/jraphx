import math
from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ops import segment_max, segment_sum

from jraphx.nn.conv import GATConv, GCNConv, SAGEConv

#: Tolerance used to guarantee that the highest scoring node of every graph
#: survives the :obj:`min_score` threshold.
_MIN_SCORE_TOL = 1e-7

#: Supported score nonlinearities.
_NONLINEARITIES = ("tanh", "sigmoid")


def _apply_nonlinearity(scores: jnp.ndarray, nonlinearity: str) -> jnp.ndarray:
    """Apply the configured score nonlinearity.

    Args:
        scores: Raw node scores of shape ``[num_nodes]``.
        nonlinearity: Name of the nonlinearity, one of :obj:`"tanh"` or :obj:`"sigmoid"`.

    Returns:
        Activated node scores of shape ``[num_nodes]``.

    Raises:
        ValueError: If ``nonlinearity`` is not a supported name.
    """
    if nonlinearity == "tanh":
        return jnp.tanh(scores)
    elif nonlinearity == "sigmoid":
        return nnx.sigmoid(scores)
    else:
        raise ValueError(
            f"Unknown nonlinearity {nonlinearity!r}, expected one of {_NONLINEARITIES}."
        )


def _segment_softmax(
    scores: jnp.ndarray, batch: jnp.ndarray | None, num_segments: int
) -> jnp.ndarray:
    """Compute a numerically stable softmax over the nodes of each graph.

    Args:
        scores: Raw node scores of shape ``[num_nodes]``.
        batch: Batch assignment vector of shape ``[num_nodes]``, or :obj:`None` for a
            single graph.
        num_segments: Number of graphs.

    Returns:
        Per-graph normalized scores of shape ``[num_nodes]``.
    """
    if batch is None:
        return nnx.softmax(scores)

    max_per_graph = segment_max(scores, batch, num_segments=num_segments)
    shifted = jnp.exp(scores - max_per_graph[batch])
    sum_per_graph = segment_sum(shifted, batch, num_segments=num_segments)
    return shifted / sum_per_graph[batch]


class TopKPooling(nnx.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

        .. math::
            \mathbf{y} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\|
            \mathbf{p} \|} \right)

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`. The softmax is taken over the nodes of each graph, and the
    highest scoring node of every graph always survives the threshold.

    .. note::
        The number of pooled nodes and edges depends on the *values* of the input, so
        this operator cannot be traced by :func:`jax.jit` or :func:`jax.vmap`. Run it
        eagerly and jit the surrounding pure-shape computations instead.

    Args:
        num_features (int): Size of each input sample.
        ratio (float or int, optional): The graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. (default: :obj:`1.0`)
        nonlinearity (str, optional): The nonlinearity to use.
            (default: :obj:`"tanh"`)
        rngs: Random number generators for initialization. Required.
    """

    def __init__(
        self,
        num_features: int,
        ratio: Union[float, int] = 0.5,
        min_score: float | None = None,
        multiplier: float = 1.0,
        nonlinearity: str = "tanh",
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            raise ValueError("TopKPooling requires `rngs` to initialize its projection vector.")

        self._init_config(
            num_features=num_features,
            ratio=ratio,
            min_score=min_score,
            multiplier=multiplier,
            nonlinearity=nonlinearity,
        )

        # Learnable scoring function
        self.weight = nnx.Param(rngs.params.uniform((1, num_features), minval=-0.01, maxval=0.01))

    def _init_config(
        self,
        num_features: int,
        ratio: Union[float, int],
        min_score: float | None,
        multiplier: float,
        nonlinearity: str,
    ) -> None:
        """Store the pooling hyperparameters shared by all top-:math:`k` style poolings.

        Args:
            num_features: Size of each input sample.
            ratio: Pooling ratio (:obj:`float`) or exact number of nodes (:obj:`int`).
            min_score: Minimal node score, or :obj:`None` to select by ``ratio``.
            multiplier: Coefficient by which features get multiplied after pooling.
            nonlinearity: Name of the score nonlinearity.

        Raises:
            ValueError: If ``nonlinearity`` is not a supported name.
        """
        if nonlinearity not in _NONLINEARITIES:
            raise ValueError(
                f"Unknown nonlinearity {nonlinearity!r}, expected one of {_NONLINEARITIES}."
            )

        self.num_features = num_features
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

    def _score(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute raw (pre-activation) node scores from the learnable projection.

        The projection :math:`\\mathbf{X}\\mathbf{p}` is divided by :math:`\\|\\mathbf{p}\\|`
        only when nodes are selected by ``ratio``. The ``min_score`` branch softmaxes the
        unscaled projection, so normalizing here would change the softmax temperature.

        Args:
            x: Node features of shape ``[num_nodes, num_features]``.
            edge_index: Edge indices of shape ``[2, num_edges]``, unused here.
            edge_attr: Edge attributes, unused here.

        Returns:
            Raw node scores of shape ``[num_nodes]``.
        """
        weight = self.weight[...]
        scores = (x * weight).sum(axis=-1)
        if self.min_score is None:
            scores = scores / jnp.linalg.norm(weight, axis=-1)
        return scores

    def _num_keep(self, num_nodes: int) -> int:
        """Number of nodes to keep for a graph of ``num_nodes`` nodes.

        Args:
            num_nodes: Number of nodes of the graph.

        Returns:
            :math:`k = \\lceil \\mathrm{ratio} \\cdot N \\rceil` for a float ratio, or
            :obj:`ratio` itself for an integer ratio, clamped to ``num_nodes``.
        """
        if isinstance(self.ratio, int):
            return min(self.ratio, num_nodes)
        return min(num_nodes, math.ceil(self.ratio * num_nodes))

    def _topk_perm(
        self, scores: jnp.ndarray, batch: jnp.ndarray | None, num_nodes: int
    ) -> jnp.ndarray:
        """Select the highest scoring nodes of every graph.

        Args:
            scores: Activated node scores of shape ``[num_nodes]``.
            batch: Batch assignment vector, or :obj:`None` for a single graph.
            num_nodes: Total number of nodes.

        Returns:
            Indices of the selected nodes, grouped by graph and sorted by decreasing
            score within each graph.
        """
        if batch is None:
            _, perm = jax.lax.top_k(scores, self._num_keep(num_nodes))
            return perm

        batch_size = int(batch.max()) + 1
        perm_list = []
        for graph_idx in range(batch_size):
            graph_mask = batch == graph_idx
            k = self._num_keep(int(graph_mask.sum()))
            if k == 0:
                continue
            graph_scores = jnp.where(graph_mask, scores, -jnp.inf)
            _, graph_perm = jax.lax.top_k(graph_scores, k)
            perm_list.append(graph_perm)

        if len(perm_list) == 0:
            return jnp.zeros((0,), dtype=jnp.int32)
        return jnp.concatenate(perm_list)

    def _select(
        self, raw_scores: jnp.ndarray, batch: jnp.ndarray | None, num_nodes: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Turn raw node scores into gating scores and selected node indices.

        Args:
            raw_scores: Raw node scores of shape ``[num_nodes]``.
            batch: Batch assignment vector, or :obj:`None` for a single graph.
            num_nodes: Total number of nodes.

        Returns:
            Tuple of the gating scores ``[num_nodes]`` and the selected node indices.
        """
        if self.min_score is None:
            scores = _apply_nonlinearity(raw_scores, self.nonlinearity)
            return scores, self._topk_perm(scores, batch, num_nodes)

        num_segments = 1 if batch is None else int(batch.max()) + 1
        scores = _segment_softmax(raw_scores, batch, num_segments)
        if batch is None:
            score_max = jnp.broadcast_to(scores.max(), scores.shape)
        else:
            score_max = segment_max(scores, batch, num_segments=num_segments)[batch]
        # Threshold on min_score, but never drop the top scoring node of a graph.
        threshold = jnp.minimum(score_max - _MIN_SCORE_TOL, self.min_score)
        perm = jnp.nonzero(scores > threshold)[0]
        return scores, perm

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        batch: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray]:
        """Apply Top-K pooling.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_features] (optional)
            batch: Batch assignment vector [num_nodes] (optional)

        Returns:
            Tuple of:
                - Pooled node features [num_pooled_nodes, num_features]
                - Pooled edge indices [2, num_pooled_edges]
                - Pooled edge attributes (optional)
                - Pooled batch assignment (optional)
                - Indices of the selected nodes [num_pooled_nodes]
        """
        num_nodes = x.shape[0]

        raw_scores = self._score(x, edge_index, edge_attr)
        scores, perm = self._select(raw_scores, batch, num_nodes)

        # Gate the selected features by their score, then apply the multiplier.
        pooled_x = x[perm] * scores[perm].reshape(-1, 1)
        if self.multiplier != 1.0:
            pooled_x = pooled_x * self.multiplier

        # Create selection mask
        mask = jnp.zeros(num_nodes, dtype=bool)
        mask = mask.at[perm].set(True)

        # Create node index mapping
        new_index = jnp.full(num_nodes, -1, dtype=jnp.int32)
        new_index = new_index.at[perm].set(jnp.arange(perm.shape[0], dtype=jnp.int32))

        # Pool edges - keep only edges between selected nodes
        row, col = edge_index[0], edge_index[1]
        edge_mask = mask[row] & mask[col]

        pooled_edge_index = jnp.stack([new_index[row[edge_mask]], new_index[col[edge_mask]]])

        # Pool edge attributes if provided
        pooled_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        # Pool batch assignment if provided
        pooled_batch = batch[perm] if batch is not None else None

        return pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, perm


class SAGPooling(TopKPooling):
    """Self-Attention Graph Pooling layer.

    From "Self-Attention Graph Pooling" (https://arxiv.org/abs/1904.08082)

    An extension of TopKPooling that uses graph convolution to compute scores,
    making them aware of the graph structure.

    .. note::
        Like :class:`TopKPooling`, the output shapes depend on the input values, so this
        operator cannot be traced by :func:`jax.jit` or :func:`jax.vmap`.

    Args:
        num_features: Number of input features
        ratio: Pooling ratio
        gnn: Type of GNN to use for score computation ('gcn', 'gat', 'sage')
        min_score: Minimum score threshold
        multiplier: Coefficient by which features get multiplied after pooling
        nonlinearity: Activation function
        rngs: Random number generators. Required.
    """

    def __init__(
        self,
        num_features: int,
        ratio: Union[float, int] = 0.5,
        gnn: str = "gcn",
        min_score: float | None = None,
        multiplier: float = 1.0,
        nonlinearity: str = "tanh",
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            raise ValueError("SAGPooling requires `rngs` to initialize its scoring GNN.")

        self._init_config(
            num_features=num_features,
            ratio=ratio,
            min_score=min_score,
            multiplier=multiplier,
            nonlinearity=nonlinearity,
        )

        self.gnn_type = gnn

        # Create GNN layer for score computation
        self.gnn: GCNConv | GATConv | SAGEConv
        if gnn == "gcn":
            self.gnn = GCNConv(num_features, 1, rngs=rngs)
        elif gnn == "gat":
            self.gnn = GATConv(num_features, 1, heads=1, rngs=rngs)
        elif gnn == "sage":
            self.gnn = SAGEConv(num_features, 1, rngs=rngs)
        else:
            raise ValueError(f"Unknown GNN type: {gnn}")

    def _score(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute raw (pre-activation) structure-aware node scores.

        Args:
            x: Node features of shape ``[num_nodes, num_features]``.
            edge_index: Edge indices of shape ``[2, num_edges]``.
            edge_attr: Edge attributes, forwarded to the scoring GNN when it is a
                :class:`~jraphx.nn.conv.GATConv`.

        Returns:
            Raw node scores of shape ``[num_nodes]``.
        """
        if isinstance(self.gnn, GATConv) and edge_attr is not None:
            scores = self.gnn(x, edge_index, edge_attr=edge_attr)
        else:
            scores = self.gnn(x, edge_index)

        return scores.squeeze(-1)
