"""
nn/mlp.py — Multi-Layer Perceptron: stacks Layer objects into a full network.

  MLP([2, 4, 4, 1])  →  input(2) → hidden(4,tanh) → hidden(4,tanh) → output(1,linear)
"""

from __future__ import annotations

from typing import Sequence

from engine.value import Value
from nn.layer     import Layer


class MLP:
    """
    A fully-connected multi-layer perceptron.

    Args:
        layer_sizes: List of integers describing the width of each layer,
                     INCLUDING the input dimension.
                     e.g. [2, 4, 4, 1] → 2 inputs, two hidden layers of 4,
                     one output neuron.

        activation:  Activation used for every hidden layer.
                     The final layer always uses 'linear' — keeps outputs
                     unbounded for regression; wrap with sigmoid/softmax in
                     the loss function for classification.

        bias:        Whether every neuron carries a bias term.

    Example:
        >>> model = MLP([2, 4, 4, 1])
        >>> out   = model([0.5, -0.3])   # returns a single Value
        >>> loss  = (out - Value(1.0)) ** 2
        >>> loss.backward()
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation:  str  = 'tanh',
        *,
        bias:        bool = True,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes needs at least [n_inputs, n_outputs], "
                f"got {layer_sizes}."
            )
        if any(s < 1 for s in layer_sizes):
            raise ValueError(f"All layer sizes must be ≥ 1, got {list(layer_sizes)}.")

        # Every hidden layer uses the chosen activation.
        # The output layer is always linear — the caller decides how to
        # interpret raw logits (MSE, BCE, cross-entropy, etc.)
        n_layers = len(layer_sizes) - 1
        self.layers: list[Layer] = [
            Layer(
                n_inputs   = layer_sizes[i],
                n_outputs  = layer_sizes[i + 1],
                activation = activation if i < n_layers - 1 else 'linear',
                bias       = bias,
            )
            for i in range(n_layers)
        ]

    # ── forward ──────────────────────────────────────────────────────────────

    def __call__(self, x: Sequence[float | Value]) -> Value | list[Value]:
        """
        Run the input through every layer in sequence.

        Each layer's output becomes the next layer's input.
        The final layer's output is returned directly.
        """
        out: list[Value] | Value = list(x)
        for layer in self.layers:
            out = layer(out)
            # Layer returns a bare Value when n_outputs == 1.
            # Wrap it back into a list so the next layer still gets a sequence.
            if isinstance(out, Value):
                out = [out]

        # Unwrap single-output networks for convenience
        return out[0] if len(out) == 1 else out

    # ── parameter access ─────────────────────────────────────────────────────

    def parameters(self) -> list[Value]:
        """Flat list of every trainable Value in the network."""
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        """Reset all gradients across the entire network."""
        for p in self.parameters():
            p.grad = 0.0

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return len(self.parameters())

    @property
    def architecture(self) -> list[tuple[int, int, str]]:
        """Returns [(n_in, n_out, activation), ...] for every layer."""
        return [
            (l.n_inputs, l.n_outputs, l._activation_name)
            for l in self.layers
        ]

    def summary(self) -> str:
        """Human-readable architecture table, similar to Keras model.summary()."""
        lines = [
            "=" * 52,
            f"{'MLP Summary':^52}",
            "=" * 52,
            f"{'Layer':<8} {'Shape':<20} {'Activation':<12} {'Params':>6}",
            "-" * 52,
        ]
        for i, layer in enumerate(self.layers):
            shape      = f"{layer.n_inputs} → {layer.n_outputs}"
            label      = "output" if i == len(self.layers) - 1 else f"hidden{i + 1}"
            lines.append(
                f"{label:<8} {shape:<20} {layer._activation_name:<12} {layer.n_params:>6}"
            )
        lines += [
            "-" * 52,
            f"{'Total trainable params':>42} {self.n_params:>6}",
            "=" * 52,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        arch = " → ".join(
            f"{l.n_inputs}[{l._activation_name}]" for l in self.layers
        ) + f" → {self.layers[-1].n_outputs}"
        return f"MLP({arch}, params={self.n_params})"