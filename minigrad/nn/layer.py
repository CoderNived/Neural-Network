"""
nn/layer.py — A fully-connected layer: n_inputs → [Neuron × n_outputs]

Sits between Neuron (single unit) and MLP (full network).
"""

from __future__ import annotations

from typing import Sequence

from engine.value import Value
from nn.neuron import Neuron


class Layer:
    """
    A single fully-connected layer: maps n_inputs → n_outputs activations.

    Every output neuron receives ALL inputs (dense connectivity).
    Each neuron has its own independent weights + bias.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: str = 'tanh',
        *,
        bias: bool = True,
    ) -> None:

        if n_inputs < 1:
            raise ValueError(
                f"n_inputs must be ≥1, got {n_inputs}"
            )

        if n_outputs < 1:
            raise ValueError(
                f"n_outputs must be ≥1, got {n_outputs}"
            )

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._activation_name = activation

        self.neurons = [
            Neuron(
                n_inputs,
                activation,
                bias=bias
            )
            for _ in range(n_outputs)
        ]

    # ─────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────

    def __call__(
        self,
        x: Sequence[float | Value]
    ) -> list[Value] | Value:

        if len(x) != self.n_inputs:
            raise ValueError(
                f"Layer expected {self.n_inputs} inputs "
                f"but got {len(x)}"
            )

        out = [n(x) for n in self.neurons]

        return out[0] if len(out) == 1 else out

    # ─────────────────────────────────────────────
    # Parameters
    # ─────────────────────────────────────────────

    def parameters(self) -> list[Value]:
        return [
            p
            for neuron in self.neurons
            for p in neuron.parameters()
        ]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    @property
    def n_params(self):
        return len(self.parameters())

    # ─────────────────────────────────────────────
    # Representation
    # ─────────────────────────────────────────────

    def __repr__(self):

        return (
            f"Layer("
            f"{self.n_inputs}→{self.n_outputs}, "
            f"activation={self._activation_name}, "
            f"params={self.n_params}"
            f")"
        )