"""Data structure for representing a single graph."""

import dataclasses
from typing import Optional

from flax.struct import dataclass
from jax import numpy as jnp


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
        custom_attr: Optional[jnp.ndarray] = None
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

    x: Optional[jnp.ndarray] = None
    edge_index: Optional[jnp.ndarray] = None
    edge_attr: Optional[jnp.ndarray] = None
    y: Optional[jnp.ndarray] = None
    pos: Optional[jnp.ndarray] = None
    batch: Optional[jnp.ndarray] = None
    ptr: Optional[jnp.ndarray] = None

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph.

        .. note::
            When inferring from edge_index, this may not be JIT-compatible
            due to dynamic shape computation.
        """
        if self.x is not None:
            return self.x.shape[0]
        elif self.edge_index is not None and self.edge_index.size > 0:
            return self.edge_index.max() + 1
        elif self.pos is not None:
            return self.pos.shape[0]
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

        A graph is undirected if for every edge (i, j), there exists an edge (j, i).
        This implementation uses vectorized operations instead of Python loops.
        """
        if self.edge_index is None or self.edge_index.shape[1] == 0:
            return False

        # Create a unique identifier for each edge using Cantor pairing
        # This avoids the need for sets and loops
        src, dst = self.edge_index[0], self.edge_index[1]

        # For undirected graphs, every edge should have its reverse
        # Create edge identifiers for both directions
        forward_edges = src * self.num_nodes + dst
        reverse_edges = dst * self.num_nodes + src

        # Check if all forward edges have corresponding reverse edges
        # Using JAX operations for efficiency
        forward_set = jnp.unique(forward_edges)

        # For each forward edge, check if reverse exists
        has_reverse = jnp.isin(reverse_edges, forward_set)

        # If all edges have their reverse, it's undirected
        return not jnp.all(has_reverse)

    def keys(self):
        """Return all attribute keys."""
        # Get all field names from the dataclass
        if dataclasses.is_dataclass(self):
            all_fields = [f.name for f in dataclasses.fields(self)]
        else:
            # Fallback to known fields
            all_fields = ["x", "edge_index", "edge_attr", "y", "pos", "batch", "ptr"]
        # Return only non-None attributes
        return [k for k in all_fields if getattr(self, k, None) is not None]

    def __contains__(self, key: str) -> bool:
        """Return True if the attribute key is present in the data."""
        return key in self.keys()

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
        return unique_nodes.size < self.num_nodes

    def has_self_loops(self) -> bool:
        """Check if the graph has self-loops.

        A self-loop is an edge from a node to itself.
        """
        if self.edge_index is None or self.edge_index.shape[1] == 0:
            return False

        # Check if any edge connects a node to itself
        src, dst = self.edge_index[0], self.edge_index[1]
        return jnp.any(src == dst)

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
