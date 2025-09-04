"""Jumping Knowledge aggregation module for JraphX."""

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
    LSTM (:obj:`"lstm"`).

    Args:
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        num_features (int, optional): The number of features per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)
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
        assert self.mode in ["cat", "max", "lstm"], f"Invalid mode: {mode}"

        if self.mode == "lstm":
            assert num_features is not None, "features cannot be None for lstm mode"
            assert num_layers is not None, "num_layers cannot be None for lstm mode"

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
            self.rnn_forward = None
            self.rnn_backward = None
            self.att = None

    def __call__(self, xs: list[jnp.ndarray]) -> jnp.ndarray:
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
                # GRUCell returns (output, new_state) tuple
                _, hidden_forward = self.rnn_forward(x[:, t, :], hidden_forward)
                forward_outputs.append(hidden_forward)

            # Backward pass through time
            for t in range(self.num_layers - 1, -1, -1):
                # GRUCell returns (output, new_state) tuple
                _, hidden_backward = self.rnn_backward(x[:, t, :], hidden_backward)
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
