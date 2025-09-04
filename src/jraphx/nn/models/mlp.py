"""Multi-Layer Perceptron (MLP) module for JraphX."""

from collections.abc import Callable
from typing import Union

import jax.numpy as jnp
from flax import nnx

from jraphx.nn.norm import BatchNorm, LayerNorm


class MLP(nnx.Module):
    r"""A Multi-Layer Perception (MLP) model.

    There exists two ways to instantiate an :class:`MLP`:

    1. By specifying explicit feature sizes, *e.g.*,

       .. code-block:: python

          mlp = MLP([16, 32, 64, 128], rngs=nnx.Rngs(0))

       creates a three-layer MLP with **differently** sized hidden layers.

    2. By specifying fixed hidden feature sizes over a number of layers,
       *e.g.*,

       .. code-block:: python

          mlp = MLP(in_features=16, hidden_features=32,
                    out_features=128, num_layers=3, rngs=nnx.Rngs(0))

       creates a three-layer MLP with **equally** sized hidden layers.

    Args:
        feature_list (List[int] or int, optional): List of input, intermediate
            and output features such that :obj:`len(feature_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_features (int, optional): Size of each input sample.
            Will override :attr:`feature_list`. (default: :obj:`None`)
        hidden_features (int, optional): Size of each hidden sample.
            Will override :attr:`feature_list`. (default: :obj:`None`)
        out_features (int, optional): Size of each output sample.
            Will override :attr:`feature_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`feature_list`. (default: :obj:`None`)
        dropout_rate (float, optional): Dropout probability of each
            hidden embedding. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to
            use. (default: :obj:`jax.nn.relu`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the module
            will not learn additive biases. (default: :obj:`True`)
        rngs: Random number generators for initialization.
    """

    def __init__(
        self,
        feature_list: Union[list[int], int, None] = None,
        *,
        in_features: int | None = None,
        hidden_features: int | None = None,
        out_features: int | None = None,
        num_layers: int | None = None,
        dropout_rate: float = 0.0,
        act: Callable | None = None,
        act_first: bool = False,
        norm: str | None = None,
        plain_last: bool = True,
        bias: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__()

        # Handle feature list construction
        if isinstance(feature_list, int):
            in_features = feature_list
            feature_list = None

        if in_features is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_features is None:
                raise ValueError(
                    f"Argument `hidden_features` must be given for `num_layers={num_layers}`"
                )
            if out_features is None:
                raise ValueError("Argument `out_features` must be given")

            feature_list = [hidden_features] * (num_layers - 1)
            feature_list = [in_features] + feature_list + [out_features]

        if feature_list is None:
            raise ValueError("Either feature_list or in_features must be specified")

        assert isinstance(feature_list, (tuple, list))
        assert len(feature_list) >= 2
        self.feature_list = list(feature_list)

        # Set activation
        self.act = act if act is not None else nnx.relu
        self.act_first = act_first
        self.plain_last = plain_last
        self.dropout_rate = dropout_rate
        self.norm_type = norm

        # Create linear layers
        self.lins = []
        for _, (in_feat, out_feat) in enumerate(
            zip(self.feature_list[:-1], self.feature_list[1:], strict=False)
        ):
            self.lins.append(
                nnx.Linear(
                    in_feat,
                    out_feat,
                    use_bias=bias,
                    rngs=rngs,
                )
            )

        # Create normalization layers
        self.norms = []
        iterator = self.feature_list[1:-1] if plain_last else self.feature_list[1:]
        for hidden_feat in iterator:
            if norm == "batch_norm":
                norm_layer = BatchNorm(hidden_feat, rngs=rngs)
            elif norm == "layer_norm":
                norm_layer = LayerNorm(hidden_feat)
            else:
                norm_layer = None
            self.norms.append(norm_layer)

        # Create dropout
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    @property
    def in_features(self) -> int:
        """Size of each input sample."""
        return self.feature_list[0]

    @property
    def out_features(self) -> int:
        """Size of each output sample."""
        return self.feature_list[-1]

    @property
    def num_layers(self) -> int:
        """Number of layers."""
        return len(self.feature_list) - 1

    def __call__(
        self,
        x: jnp.ndarray,
        batch: jnp.ndarray | None = None,
        batch_size: int | None = None,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input features [num_nodes, in_features]
            batch: Batch vector for batch normalization
            batch_size: Number of graphs in batch

        Returns:
            Output features [num_nodes, out_features]
        """
        for i, lin in enumerate(self.lins):
            x = lin(x)

            # Apply to all but last layer, or to all if not plain_last
            if i < self.num_layers - 1 or not self.plain_last:
                # Activation
                if self.act is not None and self.act_first:
                    x = self.act(x)

                # Normalization
                if i < len(self.norms) and self.norms[i] is not None:
                    norm = self.norms[i]
                    # Check if norm layer supports batch parameter
                    if self.norm_type == "batch_norm" and batch is not None:
                        x = norm(x, batch)
                    else:
                        x = norm(x)

                # Activation (if not first)
                if self.act is not None and not self.act_first:
                    x = self.act(x)

                # Dropout (not on last layer if plain_last)
                if self.dropout is not None:
                    if i < self.num_layers - 1 or not self.plain_last:
                        x = self.dropout(x)

        return x
