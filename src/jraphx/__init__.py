"""JraphX: JAX Graph Extensions - A lightweight GNN library for JAX/Flax NNX.

JraphX provides graph neural network layers and utilities for JAX,
serving as an unofficial successor to DeepMind's archived jraph library.
"""

__version__ = "0.0.1"

# Import submodules (users should import from these)
import jraphx.data
import jraphx.nn
import jraphx.utils

# Import core data structures only at top level
from jraphx.data import Batch, Data

__all__ = [
    # Core data structures
    "Data",
    "Batch",
    # Version info
    "__version__",
]
