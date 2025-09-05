"""JraphX: Graph Neural Networks with JAX/NNX.

JraphX provides graph neural network layers and utilities for JAX,
serving as an unofficial successor to DeepMind's archived jraph library.
It is derived from PyTorch Geometric code and documentation.
"""

__version__ = "0.0.2"

# Import submodules
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
