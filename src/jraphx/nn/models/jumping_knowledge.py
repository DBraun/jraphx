"""Jumping Knowledge aggregation module for JraphX."""

import jax
import jax.numpy as jnp
from flax import nnx


class JumpingKnowledge(nnx.Module):
    r"""The Jumping Knowledge layer aggregation module from the
    `"Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper.

    Jumping knowledge is performed based on either **concatenation**
    (:obj:`"cat"`)

    .. math::

        \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)},

    **max pooling** (:obj:`"max"`)

    .. math::

        \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right),

    or **weighted summation**

    .. math::

        \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

    with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
    recurrent network (:obj:`"lstm"`). The mode keeps PyG's name, but the
    recurrence is a pair of :class:`flax.nnx.GRUCell` modules run in opposite
    directions, not an LSTM, so its parameters and outputs differ from PyG's.

    Args:
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        num_features (int, optional): The number of features per representation.
            Needs to be only set for :obj:`"lstm"`-mode aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for :obj:`"lstm"`-mode aggregation. (default: :obj:`None`)
        rngs: Random number generators for initialization.
    """

    def __init__(
        self,
        mode: str,
        num_features: int | None = None,
        num_layers: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in ["cat", "max", "lstm"]:
            raise ValueError(f"Invalid mode: {mode}; expected one of 'cat', 'max', 'lstm'")

        self.features: int | None
        self.num_layers: int | None
        self.rnn_forward: nnx.GRUCell | None
        self.rnn_backward: nnx.GRUCell | None
        self.att: nnx.Linear | None

        if self.mode == "lstm":
            if num_features is None:
                raise ValueError("'num_features' is required for mode='lstm'")
            if num_layers is None:
                raise ValueError("'num_layers' is required for mode='lstm'")
            if rngs is None:
                raise ValueError(
                    "'rngs' is required for mode='lstm', which builds two GRU cells and "
                    "an attention layer"
                )

            self.features = num_features
            self.num_layers = num_layers

            # Create bidirectional LSTM using Flax NNX
            # Note: NNX doesn't have bidirectional LSTM directly, so we'll use GRU as alternative
            # Use a fixed hidden size that makes sense
            hidden_size = num_features

            # Forward and backward RNNs
            self.rnn_forward = nnx.GRUCell(
                in_features=num_features,
                hidden_features=hidden_size,
                rngs=rngs,
            )
            self.rnn_backward = nnx.GRUCell(
                in_features=num_features,
                hidden_features=hidden_size,
                rngs=rngs,
            )

            # Attention layer
            self.att = nnx.Linear(
                2 * hidden_size,  # bidirectional
                1,
                rngs=rngs,
            )
        else:
            self.features = None
            self.num_layers = None
            self.rnn_forward = nnx.data(None)
            self.rnn_backward = nnx.data(None)
            self.att = nnx.data(None)

    def __call__(self, xs: list[jax.Array]) -> jax.Array:
        """Forward pass.

        Args:
            xs: List of layer-wise representations [num_nodes, features]

        Returns:
            Aggregated representation [num_nodes, out_features]
        """
        if self.mode == "cat":
            # Concatenate along feature dimension
            return jnp.concatenate(xs, axis=-1)

        elif self.mode == "max":
            # Max pooling across layers
            stacked = jnp.stack(xs, axis=-1)  # [num_nodes, features, num_layers]
            return jnp.max(stacked, axis=-1)  # [num_nodes, features]

        else:  # self.mode == "lstm"
            # Set together with the mode in __init__; restated so that the types
            # are narrowed for the rest of this branch.
            if (
                self.rnn_forward is None
                or self.rnn_backward is None
                or self.att is None
                or self.num_layers is None
            ):
                raise RuntimeError("mode='lstm' requires the recurrent layers to be built")

            # Stack representations
            x = jnp.stack(xs, axis=1)  # [num_nodes, num_layers, features]
            num_nodes = x.shape[0]

            # Process sequences through bidirectional RNN
            # Process all nodes at once rather than using vmap to avoid module access issues

            # Initialize hidden states for all nodes
            hidden_forward = jnp.zeros((num_nodes, self.rnn_forward.hidden_features))
            hidden_backward = jnp.zeros((num_nodes, self.rnn_backward.hidden_features))

            forward_outputs = []
            backward_outputs = []

            # Forward pass through time
            for t in range(self.num_layers):
                # GRUCell takes (carry, inputs) and returns (new_carry, output)
                hidden_forward, _ = self.rnn_forward(hidden_forward, x[:, t, :])
                forward_outputs.append(hidden_forward)

            # Backward pass through time
            for t in range(self.num_layers - 1, -1, -1):
                # GRUCell takes (carry, inputs) and returns (new_carry, output)
                hidden_backward, _ = self.rnn_backward(hidden_backward, x[:, t, :])
                backward_outputs.append(hidden_backward)

            # Reverse backward outputs to match time order
            backward_outputs = backward_outputs[::-1]

            # Stack and concatenate bidirectional outputs
            # Shape: [num_layers, num_nodes, hidden_size]
            forward_stack = jnp.stack(forward_outputs, axis=0)
            backward_stack = jnp.stack(backward_outputs, axis=0)

            # Concatenate forward and backward
            # Shape: [num_nodes, num_layers, 2*hidden_size]
            bidirectional = jnp.concatenate([forward_stack, backward_stack], axis=-1)
            bidirectional = jnp.transpose(bidirectional, (1, 0, 2))

            # Compute attention weights
            alpha = self.att(bidirectional)  # [num_nodes, num_layers, 1]
            alpha = alpha.squeeze(-1)  # [num_nodes, num_layers]
            alpha = nnx.softmax(alpha, axis=-1)  # Normalize attention weights

            # Apply attention weights
            # x shape: [num_nodes, num_layers, features]
            # alpha shape: [num_nodes, num_layers]
            weighted = x * alpha[..., None]  # [num_nodes, num_layers, features]
            return jnp.sum(weighted, axis=1)  # [num_nodes, features]
