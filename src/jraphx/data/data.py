"""Data structure for representing a single graph."""

import dataclasses

from flax.struct import dataclass
from jax import numpy as jnp


def fields_equal(first: "Data", second: "Data") -> bool:
    """Compare two graph objects field by field.

    Array fields are compared element-wise and reduced to a single boolean, so
    two distinct objects holding equal arrays compare equal. A field is only
    equal to ``None`` when both operands leave it unset.

    ``flax.struct.dataclass`` regenerates ``__eq__`` for every subclass, so a
    subclass that adds fields must define ``__eq__`` in its own body and
    delegate to its base class (or here) to keep element-wise comparison.

    Args:
        first: The left-hand operand.
        second: The right-hand operand, of the same type as ``first``.

    Returns:
        True if every field of both objects holds the same value.
    """
    for field in dataclasses.fields(first):
        mine = getattr(first, field.name)
        theirs = getattr(second, field.name)
        if mine is None or theirs is None:
            if mine is not theirs:
                return False
        elif not bool(jnp.array_equal(mine, theirs)):
            return False
    return True


@dataclass
class Data:
    """A data object representing a single graph.

    This class uses flax.struct.dataclass to ensure compatibility with JAX
    transformations like jit, vmap, grad, and pmap. The Data object is
    immutable and registered as a PyTree for efficient operations.

    To add custom attributes, subclass this class:

    ```python
    @dataclass
    class MyData(Data):
        custom_attr: jnp.ndarray | None = None

        # flax.struct.dataclass regenerates __eq__ for every subclass, so
        # delegate to keep comparing array fields element-wise
        def __eq__(self, other: object) -> bool:
            return Data.__eq__(self, other)
    ```

    Attributes:
        x: Node feature matrix [num_nodes, num_features]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge feature matrix [num_edges, num_edge_features]
        y: Target labels (graph-level or node-level)
        pos: Node position matrix [num_nodes, num_dimensions]
        batch: Batch vector for batched graphs [num_nodes]
        ptr: Pointer vector for batched graphs

    Note:
        Direct attribute assignment is not supported due to immutability.
        Use the replace() method to create modified instances.
    """

    x: jnp.ndarray | None = None
    edge_index: jnp.ndarray | None = None
    edge_attr: jnp.ndarray | None = None
    y: jnp.ndarray | None = None
    pos: jnp.ndarray | None = None
    batch: jnp.ndarray | None = None
    ptr: jnp.ndarray | None = None

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph.

        The count is taken from ``x`` or ``pos`` when available, otherwise it is
        inferred as ``edge_index.max() + 1``, which undercounts isolated nodes
        whose indices exceed every edge endpoint.

        Returns:
            Number of nodes as a Python integer.

        .. note::
            The ``edge_index`` fallback reads a concrete array value and
            therefore raises under :func:`jax.jit`. Provide ``x`` or ``pos`` if
            the graph must be traced.
        """
        if self.x is not None:
            return self.x.shape[0]
        elif self.pos is not None:
            return self.pos.shape[0]
        elif self.edge_index is not None and self.edge_index.size > 0:
            return int(self.edge_index.max()) + 1
        else:
            return 0

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        if self.edge_index is not None:
            return self.edge_index.shape[1]
        else:
            return 0

    @property
    def num_node_features(self) -> int:
        """Number of node features."""
        if self.x is not None and self.x.ndim >= 2:
            return self.x.shape[-1]
        else:
            return 0

    @property
    def num_edge_features(self) -> int:
        """Number of edge features."""
        if self.edge_attr is not None and self.edge_attr.ndim >= 2:
            return self.edge_attr.shape[-1]
        else:
            return 0

    @property
    def is_directed(self) -> bool:
        """Check if the graph is directed using efficient JAX operations.

        A graph is undirected if the multiset of edges :math:`(i, j)` equals the
        multiset of reversed edges :math:`(j, i)` -- an edge appearing twice
        needs its reverse twice. The comparison sorts both endpoint tables
        lexicographically and compares them elementwise, so no node-id packing
        is involved and the check is exact at any graph size.
        """
        if self.edge_index is None or self.edge_index.shape[1] == 0:
            return False

        src, dst = self.edge_index[0], self.edge_index[1]

        def lexsorted(row: jnp.ndarray, col: jnp.ndarray) -> jnp.ndarray:
            order = jnp.lexsort((col, row))
            return jnp.stack([row[order], col[order]])

        return not bool(jnp.array_equal(lexsorted(src, dst), lexsorted(dst, src)))

    def keys(self) -> list[str]:
        """Return the names of every attribute that carries data.

        Only true dataclass fields are considered, so class-level configuration
        declared as :class:`typing.ClassVar` on subclasses is never reported.

        Returns:
            Names of the fields whose value is not ``None``.
        """
        return [f.name for f in dataclasses.fields(self) if getattr(self, f.name) is not None]

    def __contains__(self, key: str) -> bool:
        """Return True if the attribute key is present in the data."""
        return key in self.keys()

    def __eq__(self, other: object) -> bool:
        """Compare element-wise against another graph of the same type.

        Args:
            other: Object to compare against.

        Returns:
            True if ``other`` has the same type and equal field values,
            ``NotImplemented`` if the types differ so Python can try the
            reflected comparison.
        """
        if type(self) is not type(other):
            return NotImplemented
        return fields_equal(self, other)

    def has_isolated_nodes(self) -> bool:
        """Check if the graph has isolated nodes.

        A node is isolated if it doesn't appear in any edge.
        Returns False if no edges exist.
        """
        if self.edge_index is None or self.edge_index.shape[1] == 0:
            # No edges means all nodes are isolated (if any exist)
            return self.num_nodes > 0

        # Remove self-loops to check for actual connections
        edge_index = self.edge_index
        mask = edge_index[0] != edge_index[1]
        edge_index_no_loops = (
            edge_index[:, mask] if jnp.any(mask) else jnp.empty((2, 0), dtype=edge_index.dtype)
        )

        if edge_index_no_loops.shape[1] == 0:
            # Only self-loops exist, so all nodes are isolated from others
            return self.num_nodes > 0

        # Get unique nodes that appear in edges
        unique_nodes = jnp.unique(edge_index_no_loops.flatten())
        return bool(unique_nodes.size < self.num_nodes)

    def has_self_loops(self) -> bool:
        """Check if the graph has self-loops.

        A self-loop is an edge from a node to itself.
        """
        if self.edge_index is None or self.edge_index.shape[1] == 0:
            return False

        # Check if any edge connects a node to itself
        src, dst = self.edge_index[0], self.edge_index[1]
        return bool(jnp.any(src == dst))

    def __repr__(self) -> str:
        """String representation of the Data object."""
        info = []

        # Use keys() method to get all non-None attributes
        for key in self.keys():
            value = getattr(self, key)
            if hasattr(value, "shape"):
                info.append(f"{key}={list(value.shape)}")
            else:
                info.append(f"{key}={value}")

        return f"{self.__class__.__name__}({', '.join(info)})"
