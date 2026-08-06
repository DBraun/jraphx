# JraphX: Graph Neural Networks with NNX

JraphX is a graph neural network (GNN) library for [JAX](https://docs.jax.dev/), built on
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). It is an unofficial successor to
DeepMind's archived [jraph](https://github.com/google-deepmind/jraph), and its API deliberately
mirrors [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (PyG) so that models and
mental models carry over.

Everything is a plain `nnx.Module`, so layers compose with `nnx.jit`, `nnx.vmap`, `nnx.scan`,
`nnx.grad`, and the rest of the JAX ecosystem.

**What is in the box**

- **Convolutions** (`jraphx.nn.conv`): `MessagePassing`, `GCNConv`, `GATConv`, `GATv2Conv`,
  `SAGEConv`, `GINConv`, `EdgeConv`, `DynamicEdgeConv`, `TransformerConv`
- **Models** (`jraphx.nn.models`): `BasicGNN`, `GCN`, `GAT`, `GraphSAGE`, `GIN`, `MLP`,
  `JumpingKnowledge`
- **Normalization** (`jraphx.nn.norm`): `BatchNorm`, `LayerNorm`, `GraphNorm`
- **Pooling** (`jraphx.nn.pool`): `global_add_pool`, `global_mean_pool`, `global_max_pool`,
  `global_min_pool`, `global_sort_pool`, `TopKPooling`, `SAGPooling`
- **Data containers** (`jraphx.data`): `Data` and `Batch`, where batching concatenates graphs and
  offsets their node indices, exactly as in PyG
- **Utilities** (`jraphx.utils`): scatter reductions, `degree`, `add_self_loops`,
  `remove_self_loops`, `coalesce`, `to_undirected`, `to_dense_adj`, `to_edge_index`

## Installation

JraphX requires Python 3.11 or newer.

```bash
pip install jraphx
```

This pulls in JAX, Flax (0.12 or newer), and NumPy. For a source checkout:

```bash
git clone https://github.com/DBraun/jraphx.git
cd jraphx
pip install -e ".[dev]"
```

## Quick start

A two-layer GCN over a four-node cycle graph:

```python
import jax.numpy as jnp
from flax import nnx

from jraphx.data import Data
from jraphx.nn.conv import GCNConv


class GCN(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, rngs: nnx.Rngs):
        self.conv1 = GCNConv(in_features, hidden_features, rngs=rngs)
        self.conv2 = GCNConv(hidden_features, num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


data = Data(
    x=jnp.ones((4, 8)),
    edge_index=jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]]),
)

model = GCN(in_features=8, hidden_features=16, num_classes=3, rngs=nnx.Rngs(0))


@nnx.jit
def forward(model: GCN, x: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
    return model(x, edge_index)


logits = forward(model, data.x, data.edge_index)
print(logits.shape)  # (4, 3)
```

The same network is available prebuilt, with configurable depth, dropout, normalization, residual
connections, and jumping knowledge:

```python
from jraphx.nn.models import GCN

model = GCN(in_features=8, hidden_features=16, num_layers=2, out_features=3, rngs=nnx.Rngs(0))
logits = model(data.x, data.edge_index)
```

Graphs of different sizes are combined into one disjoint graph, then reduced back to per-graph
vectors by pooling against the `batch` vector:

```python
from jraphx.data import Batch
from jraphx.nn.pool import global_mean_pool

batch = Batch.from_data_list([data, data])
graph_embeddings = global_mean_pool(batch.x, batch.batch)  # (2, 8)
```

More end-to-end scripts, including Cora node classification, GAT, and GraphSAINT sampling, live in
the [`examples/`](examples) directory.

## Relationship to PyTorch Geometric

JraphX reimplements a subset of PyG on top of JAX; it does not wrap PyG and does not depend on
PyTorch. The differences you will notice when porting code:

- Layers are `nnx.Module`s and take an `rngs=nnx.Rngs(...)` argument at construction time instead of
  being initialized lazily.
- Feature dimensions are named `in_features` / `out_features` rather than PyG's
  `in_channels` / `out_channels`.
- Arrays are `jax.Array`, and `Data` / `Batch` are immutable — transformations return new objects.
- Because JAX traces static shapes, operations whose output size depends on the data (for example
  removing self-loops) are only usable outside `jax.jit`, or with a padded, fixed-size layout.
- There is no dataset or dataloader ecosystem here: bring your own pipeline (for example
  [Grain](https://github.com/google/grain)) and hand JraphX the arrays.

Layers still missing relative to PyG are tracked in the
[missing features](https://dirt.design/jraphx/missing_features.html) page.

## Documentation

Full documentation, tutorials, and the API reference: <https://dirt.design/jraphx/>

## License and attribution

JraphX is released under the Apache License 2.0 (see [LICENSE](LICENSE)).

It contains substantial portions of code and documentation derived from PyTorch Geometric
(MIT License, Copyright (c) 2023 PyG Team), and builds on Flax and DeepMind's jraph, both licensed
under the Apache License 2.0. The required third-party notices, including the MIT permission
notice, are collected in [NOTICE](NOTICE) and ship with every wheel and source distribution.
