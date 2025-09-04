from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx

from jraphx.nn.conv import GATConv, GCNConv, SAGEConv


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
    :math:`\mathbf{p}`.

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
        rngs: Random number generators for initialization.
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
        self.num_features = num_features
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        # Learnable scoring function
        self.weight = nnx.Param(rngs.params.uniform((1, num_features), minval=-0.01, maxval=0.01))

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
                - Node selection mask [num_nodes]
        """
        num_nodes = x.shape[0]

        # Compute scores for each node
        scores = (x * self.weight.value).sum(axis=-1)

        # Apply nonlinearity
        if self.nonlinearity == "tanh":
            scores = jnp.tanh(scores)
        elif self.nonlinearity == "sigmoid":
            scores = nnx.sigmoid(scores)

        # Determine number of nodes to keep
        if self.ratio < 1.0:
            if batch is None:
                k = max(1, int(self.ratio * num_nodes))
                # Get top-k indices
                _, perm = jax.lax.top_k(scores, k)
            else:
                # Handle batched graphs
                batch_size = int(batch.max()) + 1
                perm_list = []

                for b in range(batch_size):
                    batch_mask = batch == b
                    batch_scores = jnp.where(batch_mask, scores, -jnp.inf)
                    batch_num_nodes = batch_mask.sum()
                    k = max(1, int(self.ratio * batch_num_nodes))
                    k = min(k, batch_num_nodes)

                    # Get top-k for this batch
                    _, batch_perm = jax.lax.top_k(batch_scores, k)
                    perm_list.append(batch_perm)

                perm = jnp.concatenate(perm_list)
        else:
            # Keep a fixed number of nodes
            k = min(int(self.ratio), num_nodes)
            _, perm = jax.lax.top_k(scores, k)

        # Apply minimum score threshold if specified
        if self.min_score is not None:
            mask = scores[perm] > self.min_score
            perm = perm[mask]

        # Create selection mask
        mask = jnp.zeros(num_nodes, dtype=bool)
        mask = mask.at[perm].set(True)

        # Pool node features
        pooled_x = x[perm]
        pooled_scores = scores[perm].reshape(-1, 1)

        # Multiply features by scores if specified
        if self.multiplier != 1.0:
            pooled_x = pooled_x * (self.multiplier * pooled_scores)

        # Create node index mapping
        new_index = jnp.full(num_nodes, -1, dtype=jnp.int32)
        new_index = new_index.at[perm].set(jnp.arange(len(perm)))

        # Pool edges - keep only edges between selected nodes
        row, col = edge_index[0], edge_index[1]
        edge_mask = mask[row] & mask[col]

        pooled_edge_index = jnp.stack([new_index[row[edge_mask]], new_index[col[edge_mask]]])

        # Pool edge attributes if provided
        pooled_edge_attr = None
        if edge_attr is not None:
            pooled_edge_attr = edge_attr[edge_mask]

        # Pool batch assignment if provided
        pooled_batch = None
        if batch is not None:
            pooled_batch = batch[perm]

        return pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, perm


class SAGPooling(TopKPooling):
    """Self-Attention Graph Pooling layer.

    From "Self-Attention Graph Pooling" (https://arxiv.org/abs/1904.08082)

    An extension of TopKPooling that uses graph convolution to compute scores,
    making them aware of the graph structure.

    Args:
        num_features: Number of input features
        ratio: Pooling ratio
        gnn: Type of GNN to use for score computation ('gcn', 'gat', 'sage')
        min_score: Minimum score threshold
        multiplier: Score multiplier for features
        nonlinearity: Activation function
        rngs: Random number generators
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
        super().__init__(
            num_features=num_features,
            ratio=ratio,
            min_score=min_score,
            multiplier=multiplier,
            nonlinearity=nonlinearity,
            rngs=rngs,
        )

        self.gnn_type = gnn

        # Create GNN layer for score computation
        if rngs is None:
            rngs = nnx.Rngs(0)

        if gnn == "gcn":
            self.gnn = GCNConv(num_features, 1, rngs=rngs)
        elif gnn == "gat":
            self.gnn = GATConv(num_features, 1, heads=1, rngs=rngs)
        elif gnn == "sage":
            self.gnn = SAGEConv(num_features, 1, rngs=rngs)
        else:
            raise ValueError(f"Unknown GNN type: {gnn}")

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        batch: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray]:
        """Apply SAG pooling.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes (optional)
            batch: Batch assignment (optional)

        Returns:
            Same as TopKPooling
        """
        # Use GNN to compute structure-aware scores
        if self.gnn_type == "gat" and edge_attr is not None:
            scores = self.gnn(x, edge_index, edge_attr=edge_attr)
        else:
            scores = self.gnn(x, edge_index)

        scores = scores.squeeze(-1)

        # Apply nonlinearity
        if self.nonlinearity == "tanh":
            scores = jnp.tanh(scores)
        elif self.nonlinearity == "sigmoid":
            scores = nnx.sigmoid(scores)

        # Rest is same as TopKPooling but with GNN-computed scores
        num_nodes = x.shape[0]

        # Determine number of nodes to keep
        if self.ratio < 1.0:
            if batch is None:
                k = max(1, int(self.ratio * num_nodes))
                _, perm = jax.lax.top_k(scores, k)
            else:
                batch_size = int(batch.max()) + 1
                perm_list = []

                for b in range(batch_size):
                    batch_mask = batch == b
                    batch_scores = jnp.where(batch_mask, scores, -jnp.inf)
                    batch_num_nodes = batch_mask.sum()
                    k = max(1, int(self.ratio * batch_num_nodes))
                    k = min(k, batch_num_nodes)

                    _, batch_perm = jax.lax.top_k(batch_scores, k)
                    perm_list.append(batch_perm)

                perm = jnp.concatenate(perm_list)
        else:
            k = min(int(self.ratio), num_nodes)
            _, perm = jax.lax.top_k(scores, k)

        # Apply minimum score threshold
        if self.min_score is not None:
            mask = scores[perm] > self.min_score
            perm = perm[mask]

        # Create selection mask
        mask = jnp.zeros(num_nodes, dtype=bool)
        mask = mask.at[perm].set(True)

        # Pool features
        pooled_x = x[perm]
        pooled_scores = scores[perm].reshape(-1, 1)

        if self.multiplier != 1.0:
            pooled_x = pooled_x * (self.multiplier * pooled_scores)

        # Create index mapping
        new_index = jnp.full(num_nodes, -1, dtype=jnp.int32)
        new_index = new_index.at[perm].set(jnp.arange(len(perm)))

        # Pool edges
        row, col = edge_index[0], edge_index[1]
        edge_mask = mask[row] & mask[col]

        pooled_edge_index = jnp.stack([new_index[row[edge_mask]], new_index[col[edge_mask]]])

        # Pool attributes
        pooled_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
        pooled_batch = batch[perm] if batch is not None else None

        return pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, perm
