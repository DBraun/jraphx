"""Batch data structure for representing multiple graphs as a single disconnected graph."""

import sys

from flax import struct
from flax.struct import dataclass
from jax import numpy as jnp

from jraphx.data.data import Data


@dataclass
class Batch(Data):
    """A batch of graphs represented as a single large disconnected graph.

    Multiple graphs are combined by concatenating node features and
    adjusting edge indices appropriately. A batch vector tracks which
    nodes belong to which graph.

    For custom Data subclasses with additional fields, create a corresponding
    Batch subclass and optionally override class attributes to specify batching behavior:

    ```python
    from flax.struct import dataclass

    @dataclass
    class FaceData(Data):
        face: Optional[jnp.ndarray] = None
        normal: Optional[jnp.ndarray] = None
        face_color: Optional[jnp.ndarray] = None

    @dataclass
    class FaceBatch(Batch):
        face: Optional[jnp.ndarray] = None
        normal: Optional[jnp.ndarray] = None
        face_color: Optional[jnp.ndarray] = None

    # Specify which fields contain node indices that need adjustment
    FaceBatch.NODE_INDEX_FIELDS = {'face'}

    # Specify which fields should be concatenated with element-level data
    # (will be masked/filtered based on the corresponding node index field)
    FaceBatch.ELEMENT_LEVEL_FIELDS = {'normal', 'face_color'}

    # Link to corresponding Data class for proper unbatching
    FaceBatch._DATA_CLASS = FaceData
    ```
    """

    # Override these in subclasses to customize batching behavior
    NODE_INDEX_FIELDS: set[str] = struct.field(
        default_factory=set
    )  # Fields containing node indices to adjust
    ELEMENT_LEVEL_FIELDS: set[str] = struct.field(
        default_factory=set
    )  # Fields aligned with indexed element data
    GRAPH_LEVEL_FIELDS: set[str] = struct.field(
        default_factory=set
    )  # Fields that are per-graph (stacked, not concatenated)

    # Store the corresponding Data class for unbatching
    _DATA_CLASS: type = struct.field(pytree_node=False, default=None)

    @classmethod
    def from_data_list(cls, data_list: list[Data]) -> "Batch":
        """Create a batch from a list of Data objects.

        Args:
            data_list: List of Data objects to batch

        Returns:
            A Batch object containing all graphs
        """
        if len(data_list) == 0:
            return cls()

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

            elif key == "batch":
                # Skip existing batch vectors
                continue

            elif key in graph_level_fields:
                # Stack graph-level attributes
                if values:
                    batch_dict[key] = jnp.stack(values)

            elif key in element_level_fields:
                # Concatenate element-level features
                # These will be split during unbatching based on the corresponding node index field
                if all(v is not None for v in values):
                    batch_dict[key] = jnp.concatenate(values, axis=0)

            elif key in ["x", "edge_attr", "pos"]:
                # Standard node/edge features
                if all(v is not None for v in values):
                    batch_dict[key] = jnp.concatenate(values, axis=0)

            elif key == "y":
                # Handle labels based on their shape
                if len(values) > 0:
                    first_y = values[0]
                    if first_y.ndim == 0 or (first_y.ndim == 1 and first_y.shape[0] == 1):
                        # Graph-level labels
                        batch_dict["y"] = jnp.stack(values)
                    else:
                        # Node-level labels
                        batch_dict["y"] = jnp.concatenate(values, axis=0)

            else:
                # Handle unknown custom attributes with reasonable defaults
                if all(hasattr(v, "shape") for v in values):
                    # Try to concatenate tensor attributes
                    try:
                        batch_dict[key] = jnp.concatenate(values, axis=0)
                    except (ValueError, TypeError):
                        # If concat fails, try stacking
                        try:
                            batch_dict[key] = jnp.stack(values)
                        except (ValueError, TypeError):
                            batch_dict[key] = values
                else:
                    # Non-tensor attributes
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

        Returns:
            List of individual Data objects
        """
        if self.batch is None:
            return [self]

        # Get number of graphs
        num_graphs = self.batch.max() + 1

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

            # Get node mask for this graph
            node_mask = self.batch == i
            node_indices = jnp.where(node_mask)[0]

            # Extract node features
            if self.x is not None:
                data_dict["x"] = self.x[node_mask]

            if self.pos is not None:
                data_dict["pos"] = self.pos[node_mask]

            # Extract edges for this graph
            if self.edge_index is not None:
                # Find edges where both endpoints belong to this graph
                edge_mask = jnp.isin(self.edge_index[0], node_indices) & jnp.isin(
                    self.edge_index[1], node_indices
                )

                if jnp.any(edge_mask):
                    # Get edges and remap indices
                    edges = self.edge_index[:, edge_mask]

                    # Create mapping from old to new indices
                    min_idx = node_indices.min()
                    edges = edges - min_idx
                    data_dict["edge_index"] = edges

                    # Extract edge attributes
                    if self.edge_attr is not None:
                        data_dict["edge_attr"] = self.edge_attr[edge_mask]

            # Handle custom node index fields
            element_mask = None  # Will be used for face-level attributes
            for field in node_index_fields:
                if hasattr(self, field):
                    field_val = getattr(self, field)
                    if field_val is not None:
                        # Find elements where all indices belong to this graph
                        mask = jnp.all(jnp.isin(field_val, node_indices), axis=0)

                        if jnp.any(mask):
                            elements = field_val[:, mask]
                            min_idx = node_indices.min() if node_indices.size > 0 else 0
                            elements = elements - min_idx
                            data_dict[field] = elements

                            # Store mask for face-level attributes
                            if field == primary_index_field:
                                element_mask = mask

            # Handle element-level attributes using the element mask
            if element_mask is not None:
                for field in element_level_fields:
                    if hasattr(self, field):
                        field_val = getattr(self, field)
                        if field_val is not None:
                            data_dict[field] = field_val[element_mask]

            # Handle graph-level attributes
            for field in graph_level_fields:
                if hasattr(self, field):
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

        .. note::
            When inferring from batch vector, this may not be JIT-compatible
            due to dynamic computation.
        """
        if self.batch is not None:
            return self.batch.max() + 1
        elif self.ptr is not None:
            return len(self.ptr) - 1
        else:
            return 1

    def __repr__(self) -> str:
        """String representation of the Batch object."""
        info = []

        # Add batch size information if available
        num_graphs = self.num_graphs
        if num_graphs > 1:
            info.append(f"batch_size={num_graphs}")

        # Fields to exclude from repr
        exclude_fields = {
            "NODE_INDEX_FIELDS",
            "ELEMENT_LEVEL_FIELDS",
            "GRAPH_LEVEL_FIELDS",
            "_DATA_CLASS",
        }

        # Get all non-None attributes
        for field in self.__dataclass_fields__:
            if field in exclude_fields:
                continue
            value = getattr(self, field)
            if value is not None:
                if hasattr(value, "shape"):
                    info.append(f"{field}={list(value.shape)}")
                else:
                    info.append(f"{field}={value}")

        return f"{self.__class__.__name__}({', '.join(info)})"
