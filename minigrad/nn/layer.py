"""
nn/layer.py — A fully-connected layer: n_inputs → [Neuron × n_outputs]

Sits between Neuron (single unit) and MLP (full network).
"""

from __future__ import annotations

from typing import Sequence

from engine.value import Value
from nn.neuron    import Neuron


class Layer:
    """
    A single fully-connected layer: maps n_inputs → n_outputs activations.

    Every output neuron receives ALL inputs (dense connectivity).
    Each neuron has its own independent weights + bias.

    Args:
        n_inputs:   Size of the incoming feature vector.
        n_outputs:  Number of neurons (= output dimensionality).
        activation: Activation applied by every neuron in this layer.
        bias:       Whether each neuron has a learnable bias term.
    """

    def __init__(
        self,
        n_inputs:   int,
        n_outputs:  int,
        activation: str  = 'tanh',
        *,
        bias:       bool = True,
    ) -> None:
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be ≥ 1, got {n_inputs}.")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be ≥ 1, got {n_outputs}.")

        self.neurons    = [Neuron(n_inputs, activation, bias=bias) for _ in range(n_outputs)]
        self.n_inputs   = n_inputs
        self.n_outputs  = n_outputs
        self._activation_name = activation

    # ── forward ──────────────────────────────────────────────────────────────

    def __call__(self, x: Sequence[float | Value]) -> list[Value] | Value:
        """
        Forward pass through every neuron.

        Args:
            x: Input vector of length n_inputs (floats or Value nodes).

        Returns:
            List of Value nodes — one per neuron.
            If n_outputs == 1, returns the single Value directly
            (avoids callers unpacking a 1-element list every time).
        """
        if len(x) != self.n_inputs:
            raise ValueError(
                f"Layer expected {self.n_inputs} inputs, got {len(x)}."
            )

        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    # ── parameter access ─────────────────────────────────────────────────────

    def parameters(self) -> list[Value]:
        """Flat list of every trainable Value across all neurons."""
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self) -> None:
        """Reset all parameter gradients to 0.0."""
        for p in self.parameters():
            p.grad = 0.0

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return len(self.parameters())

    def __repr__(self) -> str:
        return (
            f"Layer(n_inputs={self.n_inputs}, n_outputs={self.n_outputs}, "
            f"activation={self._activation_name!r}, "
            f"params={self.n_params})"
        )