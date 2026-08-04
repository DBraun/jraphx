"""Batch data structure for representing multiple graphs as a single disconnected graph."""

import dataclasses
import sys
from typing import ClassVar

from flax.struct import dataclass
from jax import numpy as jnp

from jraphx.data.data import Data, fields_equal


def _require_all_present(key: str, values: list, num_graphs: int) -> None:
    """Reject an attribute that only some of the batched graphs carry.

    Concatenating such an attribute would produce an array whose length no
    longer matches the batch vector, silently misaligning every downstream
    scatter or gather.

    Args:
        key: Name of the attribute being collated.
        values: The non-``None`` values collected for that attribute.
        num_graphs: Total number of graphs being batched.

    Raises:
        RuntimeError: If ``values`` does not hold one entry per graph.
    """
    if len(values) != num_graphs:
        raise RuntimeError(
            f"Attribute {key!r} is present on {len(values)} of {num_graphs} graphs; "
            "batching requires it on all graphs or none"
        )


@dataclass
class Batch(Data):
    """A batch of graphs represented as a single large disconnected graph.

    Multiple graphs are combined by concatenating node features and
    adjusting edge indices appropriately. A batch vector tracks which
    nodes belong to which graph.

    Like :class:`Data`, a Batch is a pytree whose leaves are exactly its array
    attributes, so it can be passed through :func:`jax.jit`, :func:`jax.vmap`
    and :func:`jax.tree_util` transforms.

    For custom Data subclasses with additional fields, create a corresponding
    Batch subclass and override the class-level configuration to specify
    batching behavior:

    ```python
    from typing import ClassVar

    from flax.struct import dataclass

    @dataclass
    class FaceData(Data):
        face: jnp.ndarray | None = None
        normal: jnp.ndarray | None = None
        face_color: jnp.ndarray | None = None

    @dataclass
    class FaceBatch(Batch):
        face: jnp.ndarray | None = None
        normal: jnp.ndarray | None = None
        face_color: jnp.ndarray | None = None

        # Fields containing node indices that need offsetting
        NODE_INDEX_FIELDS: ClassVar[set[str]] = {"face"}

        # Fields aligned with the indexed elements, masked during unbatching
        ELEMENT_LEVEL_FIELDS: ClassVar[set[str]] = {"normal", "face_color"}

        # The Data class produced by :meth:`to_data_list`
        _DATA_CLASS: ClassVar[type | None] = FaceData
    ```
    """

    # Class-level batching configuration; override these in subclasses.
    # They are ClassVars rather than dataclass fields so that they stay out of
    # the pytree: a set is neither a valid pytree node nor hashable static
    # metadata, so declaring them as fields would break every JAX transform.
    NODE_INDEX_FIELDS: ClassVar[set[str]] = set()  # Fields containing node indices to adjust
    ELEMENT_LEVEL_FIELDS: ClassVar[set[str]] = set()  # Fields aligned with indexed element data
    GRAPH_LEVEL_FIELDS: ClassVar[set[str]] = set()  # Fields that are per-graph (stacked)

    # The corresponding Data class used when unbatching
    _DATA_CLASS: ClassVar[type | None] = None

    @classmethod
    def from_data_list(cls, data_list: list[Data]) -> "Batch":
        """Create a batch from a list of Data objects.

        Attributes are collated with torch_geometric semantics: index fields are
        offset by the running node count, fields listed in
        :attr:`GRAPH_LEVEL_FIELDS` are stacked, scalars are stacked, and
        everything else is concatenated along axis 0.

        Args:
            data_list: List of Data objects to batch.

        Returns:
            A Batch object containing all graphs.

        Raises:
            RuntimeError: If an attribute that must align with the batch vector
                is present on some but not all graphs. ``edge_index`` and the
                fields listed in :attr:`NODE_INDEX_FIELDS` are exempt, since a
                graph may legitimately have no edges or no indexed elements.
        """
        if len(data_list) == 0:
            return cls()

        num_graphs = len(data_list)

        # Collect all attributes in a dict first
        batch_dict = {}

        # Collect all attribute keys
        keys = set()
        for data in data_list:
            keys.update(data.keys())

        # Get class-level batching configuration
        node_index_fields = getattr(cls, "NODE_INDEX_FIELDS", set())
        element_level_fields = getattr(cls, "ELEMENT_LEVEL_FIELDS", set())
        graph_level_fields = getattr(cls, "GRAPH_LEVEL_FIELDS", set())

        # Process each attribute
        for key in keys:
            values = [getattr(data, key, None) for data in data_list]
            values = [v for v in values if v is not None]

            if len(values) == 0:
                continue

            if key == "edge_index":
                # Adjust edge indices for batching
                edge_indices = []
                cumsum = 0
                for data in data_list:
                    if data.edge_index is not None:
                        edge_index = data.edge_index + cumsum
                        edge_indices.append(edge_index)
                    cumsum += data.num_nodes

                if edge_indices:
                    batch_dict["edge_index"] = jnp.concatenate(edge_indices, axis=1)

            elif key in node_index_fields:
                # Handle custom node index fields (like face connectivity)
                adjusted_indices = []
                cumsum = 0
                for data in data_list:
                    val = getattr(data, key, None)
                    if val is not None:
                        adjusted_indices.append(val + cumsum)
                    cumsum += data.num_nodes

                if adjusted_indices:
                    batch_dict[key] = jnp.concatenate(adjusted_indices, axis=-1)

            elif key in ("batch", "ptr"):
                # Both are rebuilt from the node counts below
                continue

            elif key in graph_level_fields:
                # Stack graph-level attributes
                _require_all_present(key, values, num_graphs)
                batch_dict[key] = jnp.stack(values)

            elif key in element_level_fields:
                # Concatenate element-level features
                # These will be split during unbatching based on the corresponding node index field
                _require_all_present(key, values, num_graphs)
                batch_dict[key] = jnp.concatenate(values, axis=0)

            elif key in ["x", "edge_attr", "pos", "y"]:
                # Standard node/edge features and labels. Scalars are promoted
                # to a leading graph axis; everything else concatenates, so a
                # per-graph label of shape (1,) and a node-level label both end
                # up on axis 0 exactly as torch_geometric collates them.
                _require_all_present(key, values, num_graphs)
                if values[0].ndim == 0:
                    batch_dict[key] = jnp.stack(values)
                else:
                    batch_dict[key] = jnp.concatenate(values, axis=0)

            else:
                # Unknown custom attributes follow the same default collation
                _require_all_present(key, values, num_graphs)
                if all(hasattr(v, "shape") for v in values):
                    if values[0].ndim == 0:
                        batch_dict[key] = jnp.stack(values)
                    else:
                        batch_dict[key] = jnp.concatenate(values, axis=0)
                else:
                    # Non-array attributes are kept as a per-graph list
                    batch_dict[key] = values

        # Create batch vector
        batch_indices = []
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch_indices.append(jnp.full(num_nodes, i, dtype=jnp.int32))

        if batch_indices:
            batch_dict["batch"] = jnp.concatenate(batch_indices)

        # Create pointer vector (cumulative sum of nodes per graph)
        num_nodes_list = [data.num_nodes for data in data_list]
        batch_dict["ptr"] = jnp.array([0] + jnp.cumsum(jnp.array(num_nodes_list)).tolist())

        # Create Batch with all attributes at once
        return cls(**batch_dict)

    def to_data_list(self) -> list[Data]:
        """Convert batch back to a list of Data objects.

        Nodes are renumbered by their rank within their own graph, so the split
        is correct for any batch vector, including one that is not sorted. A
        graph that contributes no nodes or no edges is preserved as an empty
        graph rather than dropped.

        The distinction between a graph-level and a node-level ``y`` is
        recovered from its leading dimension, which is ambiguous when the batch
        happens to hold exactly one node per graph; in that case ``y`` is read
        as graph-level.

        Returns:
            List of individual Data objects.

        .. note::
            The number of outputs and their shapes depend on array contents, so
            this method reads concrete values and cannot be traced by
            :func:`jax.jit`.
        """
        if self.batch is None:
            return [self]

        num_graphs = self.num_graphs

        # Determine the Data class to use for unbatching
        data_class = getattr(type(self), "_DATA_CLASS", None)
        if data_class is None:
            # Fallback: try to find in same module
            if type(self).__name__.endswith("Batch"):
                data_class_name = type(self).__name__[:-5] + "Data"
                module = sys.modules[type(self).__module__]
                data_class = getattr(module, data_class_name, Data)
            else:
                data_class = Data

        # Get batching configuration
        node_index_fields = getattr(type(self), "NODE_INDEX_FIELDS", set())
        element_level_fields = getattr(type(self), "ELEMENT_LEVEL_FIELDS", set())
        graph_level_fields = getattr(type(self), "GRAPH_LEVEL_FIELDS", set())

        # Find the primary node index field for face-level data
        # (Usually the first one, like 'face' for face-level attributes)
        primary_index_field = list(node_index_fields)[0] if node_index_fields else None

        # Split data back into individual graphs
        data_list = []

        for i in range(num_graphs):
            # Collect attributes for this graph
            data_dict = {}

            # Get node mask for this graph and the rank of each node within it
            node_mask = self.batch == i
            node_map = jnp.cumsum(node_mask) - 1

            # Extract node features
            if self.x is not None:
                data_dict["x"] = self.x[node_mask]

            if self.pos is not None:
                data_dict["pos"] = self.pos[node_mask]

            # Extract edges for this graph
            if self.edge_index is not None:
                # Keep edges whose endpoints both belong to this graph
                edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
                edges = self.edge_index[:, edge_mask]
                data_dict["edge_index"] = node_map[edges].astype(self.edge_index.dtype)

                # Extract edge attributes
                if self.edge_attr is not None:
                    data_dict["edge_attr"] = self.edge_attr[edge_mask]

            # Handle custom node index fields
            element_mask = None  # Will be used for face-level attributes
            for field in node_index_fields:
                field_val = getattr(self, field)
                if field_val is not None:
                    # Keep elements whose indices all belong to this graph
                    mask = jnp.all(node_mask[field_val], axis=0)
                    elements = field_val[:, mask]
                    data_dict[field] = node_map[elements].astype(field_val.dtype)

                    # Store mask for element-level attributes
                    if field == primary_index_field:
                        element_mask = mask

            # Handle element-level attributes using the element mask
            if element_mask is not None:
                for field in element_level_fields:
                    field_val = getattr(self, field)
                    if field_val is not None:
                        data_dict[field] = field_val[element_mask]

            # Handle graph-level attributes
            for field in graph_level_fields:
                field_val = getattr(self, field)
                if field_val is not None:
                    data_dict[field] = field_val[i]

            # Handle labels
            if self.y is not None:
                if self.y.shape[0] == num_graphs:
                    # Graph-level labels
                    data_dict["y"] = self.y[i]
                else:
                    # Node-level labels
                    data_dict["y"] = self.y[node_mask]

            # Create Data object with all attributes
            data = data_class(**data_dict)
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self) -> int:
        """Number of graphs in the batch.

        ``ptr`` is preferred because it records one entry per graph and can
        therefore represent a graph with zero nodes, which the batch vector
        cannot.

        Returns:
            Number of graphs as a Python integer.

        .. note::
            The ``batch`` fallback reads a concrete array value and therefore
            raises under :func:`jax.jit`.
        """
        if self.ptr is not None:
            return len(self.ptr) - 1
        elif self.batch is not None:
            return int(self.batch.max()) + 1
        else:
            return 1

    def __eq__(self, other: object) -> bool:
        """Compare element-wise against another batch of the same type.

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

    def __repr__(self) -> str:
        """String representation of the Batch object."""
        info = []

        # Add batch size information if available
        num_graphs = self.num_graphs
        if num_graphs > 1:
            info.append(f"batch_size={num_graphs}")

        # Get all non-None attributes
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                if hasattr(value, "shape"):
                    info.append(f"{field.name}={list(value.shape)}")
                else:
                    info.append(f"{field.name}={value}")

        return f"{self.__class__.__name__}({', '.join(info)})"
